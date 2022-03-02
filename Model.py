#!/opt/anaconda3/bin/python

"""
This script contains the architecture of the neural network. 
Model inspired from:
`"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding " <https://arxiv.org/pdf/2106.11170.pdf>`.

Usage: type "from Model import <class>" to use one of its classes.
       type "from Model import <function>" to use one of its functions.

Contributors: Ambroise Odonnat.
"""

import torch

import torch.nn.functional as F
import numpy as np

from torch import nn
from torch import Tensor
from torch.autograd import Variable

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce



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

class channel_attention(nn.Module):
    
    def __init__(self, linear_size, vector_size, dropout, negative_slope, kernel_size, stride):
        
        """    
        Args:
            linear_size (int): Size of input and output in linear layer,
            scaling (int): Scaling in scaled dot-product attention ,
            dropout (float): Value of dropout,
            negative_slope (float): Slope in Leaky Relu,
            kernel_size (int): Kernel size in average pooling layer,
            stride (int): Value of stride in average pooling layer.
        """
        
        super().__init__()
        self.scaling = vector_size ** (1 / 2)

        # Query layer
        self.query = nn.Sequential(
            nn.Linear(linear_size, linear_size),
            nn.LayerNorm(linear_size),  # also may introduce improvement to a certain extent
            nn.Dropout(dropout)
        )
        
        # Key layer
        self.key = nn.Sequential(
            nn.Linear(linear_size, linear_size),
            # nn.LeakyReLU(),
            nn.LayerNorm(linear_size),
            nn.Dropout(dropout)
        )

        # Final projection layer
        self.projection = nn.Sequential(
            nn.Linear(linear_size, linear_size),
            nn.LeakyReLU(negative_slope),
            nn.LayerNorm(linear_size),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(0)
        
        # Average pooling layer
        self.pooling = nn.AvgPool2d(kernel_size=(1, kernel_size), stride=(1, stride))

        # Weights initiation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):

        """
        Apply spatial transforming, i.e attention score, on input batch x of size (batch_size x 1 x n_channels x n_sample_points).

        Args:
            x (torch tensor): Batch of trial after spatial filtering of dimension (batch_size x 1 x n_channels x n_sample_points).

        Returns:
            Attention weighted data: torch tensor of same dimension as x (batch_size x 1 x n_channels x n_sample_points).
        """
        
        temp = rearrange(x, 'b o c t -> b o t c')

        # Compute query Q
        temp_query = rearrange(self.query(temp), 'b o t c -> b o c t')
        channel_query = self.pooling(temp_query)
        
        # Compute key K
        temp_key = rearrange(self.key(temp), 'b o t c -> b o c t')
        channel_key = self.pooling(temp_key)

        # Compute attention score
        channel_atten = torch.einsum('b o c t, b o m t -> b o c m', channel_query, channel_key) / self.scaling
        channel_atten_score = F.softmax(channel_atten, dim=-1)
        channel_atten_score = self.dropout(channel_atten_score)

        # Multiply by value V = x
        out = torch.einsum('b o c t, b o c m -> b o c t', x, channel_atten_score)

        # Apply projection
        out = rearrange(out, 'b o c t -> b o t c')
        out = self.projection(out)
        out = rearrange(out, 'b o t c -> b o c t')
        return out
    
    

""" ********** Position encoding and compression ********** """

class PatchEmbedding(nn.Module):
    
    def __init__(self, out_channels, position_kernel_size, position_stride, negative_slope,\
                 channel_kernel_size, time_kernel_size, time_stride, slice_size):
        
        """    
        Args:
            out_channels (int): Number of output channels in convolutional layer for position encoding,
            position_kernel_size (int): Kernel size for position encoding on time axis,
            position_stride (float): Stride for position encoding on time axis,
            negative_slope (int): Slope in Leaky Relu,
            channel_kernel_size (int): Kernel size in convolutional layer on channel axis,
            time_kernel_size (int): Kernel size in convolutional layer on time axis,
            time_stride (int): Stride in convolutional layer on channel axis,
            slice_size (int): Number of output channels in convolutional layer for slicing.
        """
        
        super().__init__()
        
        # Position encoding and compression of channel axis via convolutional layer
        self.projection = nn.Sequential(  
            
            # Position encoding
            nn.Conv2d(1, out_channels, (1, position_kernel_size), (1, position_stride)),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope),
            
            # Slicing and compressing on channel axis
            nn.Conv2d(out_channels, slice_size,\
                      (channel_kernel_size, time_kernel_size), stride=(1, time_stride)),
            Rearrange('b o (c) (t) -> b (c t) o'),
        )

    def forward(self, x: Tensor) -> Tensor:

        """
        Compress and slice input batch x of size (batch_size x 1 x n_channels x n_sample_points).

        Args:
            x (torch tensor): Batch of trial after spatial transforming of dimension (batch_size x 1 x n_channels x n_sample_points).

        Returns:
            x: Compressed and sliced batch: torch tensor of size (batch_size x new_n_sample_points x slice_size).
        """
        
        # Apply projection
        x = self.projection(x)
        return x
    

    
""" ********** Multi-head attention ********** """

class MultiHeadAttention(nn.Module):
    
    def __init__(self, slice_size, num_heads, dropout):
        
        """    
        Args:
            slice_size (int): Size of slices,
            num_heads (int): Number of heads in mulit-head block,
            dropout (float): Dropout value for multi-head block,.
        """
        
        super().__init__()
        self.scaling = slice_size ** (1 / 2)
        self.num_heads = num_heads
        self.keys = nn.Linear(slice_size, slice_size)
        self.queries = nn.Linear(slice_size, slice_size)
        self.values = nn.Linear(slice_size, slice_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(slice_size, slice_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
       
        """
        Apply mulit-head attention on compressed and sliced batch x of size (batch_size x new_n_sample_points x slice_size).

        Args:
            x (torch tensor): Batch of trial after compression and slicing (batch_size x new_n_sample_points x slice_size).

        Returns:
             out: Torch tensor of size (batch_size x new_n_sample_points x slice_size).
        """
        
        # Compute Q, K, V
        queries = rearrange(self.queries(x), 'b q (h o) -> b h q o', h = self.num_heads) # q = query length
        keys = rearrange(self.keys(x), 'b k (h o) -> b h k o', h = self.num_heads) # k = key length
        values = rearrange(self.values(x), 'b v (h o) -> b h v o', h = self.num_heads) # v = value length and q = k = v
        
        # Compute attention score
        energy = torch.einsum('b h q o, b h k o -> b h q k', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        attention_score = F.softmax(energy / self.scaling, dim=-1)
        attention_score = self.att_drop(attention_score)
        
        # Multiply by value V 
        out = torch.einsum('b h q k, b h v o -> b h v o ', attention_score, values) # q = v
        
        # Apply projection
        out = rearrange(out, "b h v o -> b v (h o)")
        out = self.projection(out)
        return out
    
    
    
""" ********** Feed-forward block ********** """

class GELU(nn.Module):
    
    """
    GeLU activation function.
    """
    
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))
    

class FeedForwardBlock(nn.Sequential):
    
    def __init__(self, slice_size, size_expansion, dropout):
        
        """    
        Args:
            slice_size (int): Size of slices,
            size_expansion (int): Value of expansion to obtain inner size,
            dropout (float): Dropout value.
        """
        
        super().__init__(
            nn.Linear(slice_size, size_expansion * slice_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size_expansion * slice_size, slice_size),
        )
        
        
        
""" ********** Temporal transforming ********** """

class TransformerEncoderBlock(nn.Sequential):
    
    def __init__(self, slice_size, num_heads, dropout, forward_expansion, forward_dropout):

        """    
        Args:
            slice_size (int): Size of slices,
            num_heads (int): Number of heads in mulit-head block,
            dropout (float): Dropout value for multi-head block,
            forward_expansion (int): Value of expansion to obtain inner size,
            forward_dropout (float): Dropout value for FF block.
        """
        
        super().__init__(
            
            # Multi-head attention
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(slice_size),
                MultiHeadAttention(slice_size, num_heads, dropout),
                nn.Dropout(dropout)
            )),
            
            # Feed-forward block
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(slice_size),
                FeedForwardBlock(slice_size, forward_expansion, forward_dropout),
                nn.Dropout(dropout)
            )
            ))

        
class TransformerEncoder(nn.Sequential):
    
    def __init__(self, depth, slice_size, num_heads, dropout, forward_expansion, forward_dropout):

        """    
        Args:
            depth (int): Number of {multi-head attention + FF} blocks,
            slice_size (int): Size of slices,
            num_heads (int): Number of heads in mulit-head block,
            dropout (float): Dropout value for multi-head block,
            forward_expansion (int): Value of expansion to obtain inner size,
            forward_dropout (float): Dropout value for FF block.
        """
        
        super().__init__(*[TransformerEncoderBlock(slice_size, num_heads, dropout,\
                                                   forward_expansion, forward_dropout) for _ in range(depth)])
        

        
""" ********** Classifier ********** """

class ClassificationHead(nn.Sequential):
    
    def __init__(self, slice_size, n_classes):
    
        """    
        Args:
            slice_size (int): Size of slices,
            n_classes (int): Number of classes.
        """
        
        super().__init__()
        
        self.clshead = nn.Sequential(
            Reduce('b v o -> b o', reduction='mean'),
            nn.LayerNorm(slice_size),
            nn.Linear(slice_size, n_classes)
        )

    def forward(self, x):
        
        """
        Compress and slice input batch x of size (batch_size x 1 x n_channels x n_sample_points).

        Args:
            x (torch tensor): Batch of trial after transformer of dimension (batch_size x new_n_sample_points x slice_size).

        Returns:
            tuple: x : Batch of trial after transformer of dimension (batch_size x new_n_sample_points x slice_size),
                   out: Vector of probability of dimension (batch_size x n_classes) 
        """
        
        out = self.clshead(x)
        return x, out

    

""" ********** Model ********** """

class Transformer(nn.Sequential):
    
    """ 
    Transformer model based on:
    `"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding " <https://arxiv.org/pdf/2106.11170.pdf>`.
    
    Input is of dimension (batch_size x 1 x n_channels x n_sample_points).
    Output is of dimension (batch_size x new_sample_points x slice_size).
    """
    
    def __init__(self, normalized_shape = 201, linear_size = 28, vector_size = 201,\
                 attention_dropout = 0.3, attention_negative_slope = 1e-2, attention_kernel_size = 30, attention_stride = 30,\
                 spatial_dropout = 0.5,\
                 out_channels = 2, position_kernel_size = 51, position_stride = 1, emb_negative_slope = 0.2,\
                 channel_kernel_size = 28, time_kernel_size = 5, time_stride = 5, slice_size = 10,\
                 depth = 3, num_heads = 5, transformer_dropout = 0.5, forward_expansion = 4, forward_dropout = 0.5,\
                 n_classes = 7): #, **kwargs):
      
        """    
        Args:
            slice_size (int): Size of slices,
            size_expansion (int): Value of expansion to obtain inner size,
            drop_p (float): Dropout value.
        """
        
        super().__init__(   
            
            # Spatial transforming,
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(normalized_shape),
                    channel_attention(linear_size, vector_size, attention_dropout, attention_negative_slope,\
                                      attention_kernel_size, attention_stride),
                    nn.Dropout(spatial_dropout),
                )
            ),
            
            # Position encoding, compression and slicing
            PatchEmbedding(out_channels, position_kernel_size, position_stride, emb_negative_slope,\
                 channel_kernel_size, time_kernel_size, time_stride, slice_size),
            
            # Temporal transforming
            TransformerEncoder(depth, slice_size, num_heads, transformer_dropout, forward_expansion, forward_dropout),
            
            # Classifier
            ClassificationHead(slice_size, n_classes)
        )