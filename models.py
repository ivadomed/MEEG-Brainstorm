#!/usr/bin/env python

"""
This script contains a model to detect spikes and a model to count the number of spikes inspired by:
`"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding" <https://arxiv.org/pdf/2106.11170.pdf>`_.

Usage: type "from models import <class>" to use one of its classes.

Contributors: Ambroise Odonnat.
"""

import sys
import torch

import numpy as np
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce
from torch import nn
from torch import Tensor
from torch.autograd import Variable

from heads import Mish, RobertaClassifier, SpikeDetector
from time_embedding import Time2Vec


""" ********** Residual connection for better training ********** """

class ResidualAdd(nn.Module):
    
    def __init__(self, fn):
        
        """    
        Args:
            fn: Sequence of layers.
        """
        
        super().__init__()
        
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


    
""" ********** Spatial transforming ********** """

class ChannelAttention(nn.Module):
    
    def __init__(self, emb_size, num_heads, dropout):
        
        """    
        Multi-head attention inspired by:
        `"Attention Is All You Need" <https://arxiv.org/pdf/1606.08415v3.pdf>`_.
        Trials were padded with zero channels to have the same sequence dimension in a given batch.
        Args:
            emb_size (int): Size of embedding vectors. (! Warning: num_heads must be a dividor of emb_size !).
            num_heads (int): Number of heads in multi-head block.
            dropout (float): Dropout value in multi-head block.
        """
        
        super().__init__()
        
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout)

    def forward(self, x):

        """
        Apply spatial transforming.

        Args:
            x (torch tensor): Batches of trials of dimension [batch_size x 1 x n_channels x n_time_points].

        Returns:
            out (tensor): Batches of trials of dimension [batch_size x 1 x n_channels x n_time_points].
        """
        
        # Multi-head attention
        x = torch.squeeze(x)
        #key_padding_mask = (x.mean(dim=-1) == -sys.maxsize)&(x.std(dim=-1) == 0) # padded channels should be ignored in self-attention 
        x = rearrange(x, 'b s e -> s b e')
        out, _ = self.attention(x, x, x)
        out = rearrange(out, 's b e -> b s e')
        out = torch.unsqueeze(out,1)
        return out

    

""" ********** Input embedding and positional encoding with convolutions ********** """

class PatchEmbedding(nn.Module):
    
    def __init__(self, padding, seq_len, position_kernel, position_stride, emb_size, n_channels, time_kernel, time_stride):
        
        """    
        Args:
            padding (bool): If True, apply padding.
            seq_len (int): Width of input x (corresponds to the number of time points in the EEG/MEG trial).
            position_kernel (int): Kernel size for position encoding on time axis.
            position_stride (float): Stride for position encoding on time axis. 
            emb_size (int): Number of output channels in convolutional layer for slicing.
            n_channels (int): Number of channels after CSP Projection.
            time_kernel (int): Kernel size in convolutional layer on time axis.
            time_stride (int): Stride in convolutional layer on channel axis.
        """
        
        super().__init__()
        
        if padding:
            
            # Padding values to preserve seq_len
            position_padding = int(((position_stride-1) * seq_len - (position_stride+1) + position_kernel) / 2)
            seq_len = int((((seq_len + 1 - position_stride) / position_stride) + 1) / 2) 
            time_padding = int(((time_stride-1) * seq_len - (time_stride+1) + time_kernel) / 2) + 1

            # Position encoding and compression of channel axis via convolutional layer
            self.embedding = nn.Sequential(nn.Conv2d(1, 2, (1, position_kernel),
                                                     stride=(1, position_stride),
                                                     padding=(0, position_padding)), 
                                           nn.BatchNorm2d(2),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(2, emb_size, (n_channels, time_kernel),
                                                     stride=(1, time_stride),
                                                     padding=(0,time_padding)), 
                                           Rearrange('b o (c) (t) -> b (c t) o')) # [batch size x seq_len x emb_size]   
        else:
            
            # Position encoding and compression of channel axis via convolutional layer
            self.embedding = nn.Sequential(nn.Conv2d(1, 2, (1, position_kernel),
                                                     stride=(1, position_stride)),
                                           nn.BatchNorm2d(2),
                                           nn.LeakyReLU(),
                                           nn.Conv2d(2, emb_size, (n_channels, time_kernel),
                                                     stride=(1, time_stride)), 
                                           Rearrange('b o (c) (t) -> b (c t) o')) # [batch size x new_seq_len x emb_size]
        

    def forward(self, x: Tensor):

        """
        Create embeddings with positional encoding. 

        Args:
            x (tensor): Batch of trials after spatial transforming of dimension [batch_size x 1 x n_channels x seq_len].

        Returns:
            x (tensor): Batches of embeddings of dimension [batch_size x new_seq_len x emb_size].
                        Warning: new_seq_len might differ from seq_len due to convolutional layers.
                        If padding, x has dimension [batch_size x seq_len x emb_size].
        """
        
        # Create embeddings with positional encoding
        x = self.embedding(x)
        return x
    
    

""" ********** Input embedding and positional encoding with Time2Vec ********** """

class TimeEmbedding(nn.Module):
    
    def __init__(self, seq_len, emb_size, n_channels):
        
        """  
        Apply Time2Vec method inspired by:
        `"Time2Vec: Learning a Vector Representation of Time" <https://arxiv.org/pdf/1907.05321.pdf>`_.
        
        Args:
            seq_len (int): Length of sequence (corresponds to the number of time points in the EEG/MEG trial).
            emb_size (int): Size of embedding vectors.
            n_channels (int): Number of channels after CSP projection.
        """
        
        super().__init__()
        
        self.embedding = Time2Vec(seq_len, emb_size, n_channels)

    def forward(self, x: Tensor):

        """
        Args:
            x (torch tensor): Batch of trial after spatial transforming of dimension [batch_size x 1 x n_channels x seq_len].

        Returns:
            x (tensor): Batch with positional encoding of dimension [batch_size x seq_len x emb_size].
        """
        
        embedding = self.embedding(x)
        
        return embedding

    
    
""" ********** Multi-head attention ********** """

class MultiHeadAttention(nn.Module):
    
    def __init__(self, emb_size, num_heads, dropout):
        
        """   
        Multi-head attention inspired by:
        `"Attention Is All You Need" <https://arxiv.org/pdf/1606.08415v3.pdf>`_.
        Args:
            emb_size (int): Size of embedding vectors. (! Warning: num_heads must be a dividor of emb_size !).
            num_heads (int): Number of heads in multi-head block.
            dropout (float): Dropout value in multi-head block.
        """
        
        super().__init__()
        
        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout)

    def forward(self, x: Tensor):
       
        """
        Apply multi-head attention.

        Args:
            x (tensor): Batch of trials with dimension [batch_size x seq_len x emb_size].

        Returns:
             out: Batch of trials with dimension [batch_size x seq_len x emb_size].
        """
        x = rearrange(x, 'b s e -> s b e')
        out, _ = self.attention(x, x, x)
        out = rearrange(out, 's b e -> b s e')

        return out    
    
    
    
""" ********** Feed-forward block ********** """

class FeedForwardBlock(nn.Sequential):
    
    def __init__(self, emb_size, expansion, dropout):
        
        """    
        Novel method: Feed Forward with Mish activation instead of GELU activation.
        Inspired by: `"Attention Is All You Need" <https://arxiv.org/pdf/1606.08415v3.pdf>`_.
        
        Args:
            emb_size (int): Size of embedding vectors.
            expansion (int): Expansion coefficient to obtain inner size.
            dropout (float): Dropout value.
        """
        
        super().__init__(nn.Linear(emb_size, expansion * emb_size),
                         Mish(),
                         nn.Dropout(dropout),
                         nn.Linear(expansion * emb_size, emb_size))
        
        
        
""" ********** Temporal transforming ********** """

class TransformerEncoderBlock(nn.Sequential):
    
    def __init__(self, emb_size, num_heads, expansion, dropout):

        """    
        Args:
            emb_size (int): Size of embedding vectors.
            num_heads (int): Number of heads in multi-head block.
            expansion (int): Expansion coefficient to obtain inner size in FF block.
            dropout (float): Dropout value.
        """
        
        super().__init__(ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size),
                                                   MultiHeadAttention(emb_size, num_heads, dropout),
                                                   nn.Dropout(dropout))), # MHA block
                         
                         ResidualAdd(nn.Sequential(nn.LayerNorm(emb_size),
                                                   FeedForwardBlock(emb_size, expansion, dropout),
                                                   nn.Dropout(dropout)))) # Novel Feed forward block

        
        
class TransformerEncoder(nn.Sequential):
    
    def __init__(self, depth, emb_size, num_heads, expansion, dropout):

        """    
        Args:
            depth (int): Number of Transformer layers.
            emb_size (int): Size of embedding vectors.
            num_heads (int): Number of heads in multi-head block.
            expansion (int): Expansion coefficient to obtain inner size in FF block.
            forward_dropout (float): Dropout value.
        """
        
        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads, expansion, dropout) for _ in range(depth)])

    

""" ********** Model ********** """

class ClassificationBertMEEG(nn.Sequential):
    
    """ 
    Determine the number of spikes in an EEG/MEG trial. Inspired by:
    `"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding" <https://arxiv.org/pdf/2106.11170.pdf>`_.
    
    Input (tensor): Tensor of trials after CSP projection of dimension [batch_size x 1 x n_channels x n_time_points].
    Output (tensor): Tensor of logits of dimension [batch_size x n_classes].
    """
    
    def __init__(self, n_classes, n_channels, n_time_points, attention_dropout, attention_kernel, attention_stride,
                 spatial_dropout, position_kernel, position_stride, emb_size, time_kernel, time_stride,
                 embedding_dropout, depth, num_heads,  expansion, transformer_dropout, classifier_dropout): 
      
        """    
        Args:
            n_classes (int): Number of classes in the dataset.
            n_channels (int): Number of channels in EEG/MEG trials after CSP Projection.
            n_time_points (int): Number of time points in EEF/MEG trials.
            attention_dropout (float): Dropout value in channel_attention layer.
            attention_kernel (int): Average pooling kernel size in channel_attention layer.
            attention_stride (int): Average pooling stride in channel_attention layer.
            spatial_dropout (float): Dropout value after Spatial transforming block.
            position_kernel (int): Kernel size in convolution for positional encoding.
            position_stride (int): Stride in convolution for positional encoding.
            emb_size (int): Size of embedding vectors in Temporal transforming block.
            time_kernel (int): Kernel size in convolution for embedding.
            time_stride (int): Stride in convolution for embedding.
            embedding_dropout (float): Dropout value after embedding block.
            depth (int): Depth of the Transformer encoder.
            num_heads (int): Number of heads in multi-attention layer (! Warning: num_heads must be a dividor of emb_size !).
            expansion (int): Expansion coefficient in Feed Forward layer.
            transformer_dropout (float): Dropout value in Temporal transforming block.
            classifier_dropout (float): Dropout value in Classifier block.
        """
        
        super().__init__(ResidualAdd(nn.Sequential(nn.LayerNorm(n_time_points),
                                                   ChannelAttention(n_channels, n_time_points,
                                                                    attention_dropout,
                                                                    attention_kernel,
                                                                    attention_stride),
                                                   nn.Dropout(spatial_dropout))), # Spatial transforming
                         
                         PatchEmbedding(True, n_time_points, position_kernel, position_stride,
                                        emb_size, n_channels, time_kernel, time_stride), # Embedding and positional encoding
                         
                         nn.Dropout(embedding_dropout),
                         
                         TransformerEncoder(depth, emb_size, num_heads, expansion, transformer_dropout), # Temporal transforming

                         RobertaClassifier(emb_size, n_classes, classifier_dropout)) # Classifier
        

        
class DetectionBertMEEG(nn.Sequential):
    
    """ 
    Detect spikes events times in an EEG/MEG trial. Inspired by:
    `"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding" <https://arxiv.org/pdf/2106.11170.pdf>`_.
    
    Input (tensor): Tensor of trials after CSP projection of dimension [batch_size x 1 x n_channels x n_time_points].
    Output (tensor): Tensor of logits of dimension [batch_size x n_time_windows x 2].
    """
    
    def __init__(self, n_channels, n_time_points, attention_num_heads, attention_dropout, spatial_dropout,
                 position_kernel, position_stride, emb_size, time_kernel, time_stride,embedding_dropout, depth, num_heads,
                 expansion, transformer_dropout, n_time_windows, detector_dropout): 
      
        """    
        Args:
            n_channels (int): Number of channels in EEG/MEG trials after CSP Projection.
            n_time_points (int): Number of time points in EEF/MEG trials.
            attention_num_heads (int): Number of heads in channel attention layer.
            attention_dropout (float): Dropout value in channel_attention layer.
            position_kernel (int): Kernel size in convolution for positional encoding.
            position_stride (int): Stride in convolution for positional encoding.
            emb_size (int): Size of embedding vectors in Temporal transforming block.
            time_kernel (int): Kernel size in convolution for embedding.
            time_stride (int): Stride in convolution for embedding.
            embedding_dropout (float): Dropout value after embedding block.
            depth (int): Depth of the Transformer encoder.
            num_heads (int): Number of heads in multi-attention layer.
            expansion (int): Expansion coefficient in Feed Forward layer.
            transformer_dropout (float): Dropout value after Temporal transforming block.
            n_time_windows (int): Number of time windows.
            detector_dropout (float): Dropout value in spike detector block.
        """
        
        super().__init__(ResidualAdd(nn.Sequential(nn.LayerNorm(n_time_points),
                                                   ChannelAttention(n_time_points,
                                                                    attention_num_heads,
                                                                    attention_dropout),
                                                   nn.Dropout(spatial_dropout))), # Spatial transforming,

                         PatchEmbedding(True, n_time_points, position_kernel, position_stride,
                                        emb_size, n_channels, time_kernel, time_stride), # Embedding and positional encoding

                         nn.Dropout(embedding_dropout),

                         TransformerEncoder(depth, emb_size, num_heads, expansion, transformer_dropout), # Temporal transforming

                         SpikeDetector(n_time_points, n_time_windows, emb_size, detector_dropout)) # Spike detector

