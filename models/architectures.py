#!/usr/bin/env python

"""
This script contains a model to detect spikes and a model
to count the number of spikes inspired by:
`"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
<https://arxiv.org/pdf/2106.11170.pdf>`_.

Usage: type "from models import <class>" to use one class.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import torch

import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch import Tensor
from utils.utils_ import normal_initialization, xavier_initialization


""" ********** Mish activation ********** """


class Mish(nn.Module):

    """ Activation function inspired by:
        `<https://www.bmvc2020-conference.com/assets/papers/0928.pdf>`.
    """

    def __init__(self):

        super().__init__()

    def forward(self,
                x: Tensor):

        return x*torch.tanh(F.softplus(x))


""" ********** Spatial transforming ********** """


class ChannelAttention(nn.Module):

    def __init__(self,
                 emb_size,
                 num_heads,
                 dropout):

        """ Multi-head attention inspired by:
            `"Attention Is All You Need"
            <https://arxiv.org/pdf/1606.08415v3.pdf>`_.

        Args:
            emb_size (int): Size of embedding vectors (here: n_time_points).
                            Warning -> num_heads must be a
                                       dividor of emb_size !
            num_heads (int): Number of heads in multi-head block.
            dropout (float): Dropout value.
        """

        super().__init__()

        self.attention = nn.MultiheadAttention(emb_size, num_heads,
                                               dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_size)

        # Weight initialization
        self.attention.apply(xavier_initialization)

    def forward(self,
                x: Tensor):

        """ Apply spatial transforming.
            Trials can be padded with zeros channels for same sequence length.

        Args:
            x (torch tensor): Batches of trials of dimension
                              [batch_size x 1 x n_channels x n_time_points].

        Returns:
            out (tensor): Batches of trials of dimension
                          [batch_size x 1 x n_channels x n_time_points].
        """

        temp = torch.squeeze(x, dim=1)

        # padded channels are ignored in self-attention
        mask = (temp.mean(dim=-1) == 0) & (temp.std(dim=-1) == 0)
        temp = rearrange(temp, 'b s e -> s b e')
        temp, attention_weights = self.attention(temp, temp, temp,
                                                 key_padding_mask=mask)
        temp = rearrange(temp, 's b e -> b s e')
        x_attention = self.dropout(temp).unsqueeze(1)
        out = self.norm(x + x_attention)

        return out, attention_weights


""" ********** Embedding and positional encoding ********** """


class PatchEmbedding(nn.Module):

    def __init__(self,
                 seq_len,
                 emb_size,
                 n_maps,
                 position_kernel,
                 channels_kernel,
                 channels_stride,
                 time_kernel,
                 time_stride,
                 dropout):

        """Positional encoding and embedding. Inspired by:
            `"EEGNet: a compact convolutional neural network for EEG-based
            brain–computer interfaces"
            <https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/pdf>`_.

        Args:
            seq_len (int): Sequence length (here: n_time_points).
            emb_size (int): Size of embedding vectors.
            n_maps (int): Number of feature maps for positional encoding.
            position_kernel (int): Kernel size for positional encoding.
            channels_kernel (int): Kernel size for convolution on channels.
            channels_stride (int): Stride for convolution on channels.
            time_kernel (int): Kernel size for convolution on time axis.
            time_stride (int): Stride for convolution on channel axis.
            dropout (float): Dropout value.
        """

        super().__init__()

        # Padding values to preserve seq_len
        position_padding = position_kernel-1
        position_padding = int(position_padding / 2) + 1
        new_seq_len = int(seq_len + 2*position_padding
                          - position_kernel + 1)
        time_padding = ((time_stride-1) * new_seq_len
                        + time_kernel) - time_stride
        if (time_kernel % 2 == 0) & (time_stride % 2 == 0):
            time_padding = int(time_padding / 2) - 1
        elif (time_kernel % 2 != 0) & (time_stride % 2 != 0):
            time_padding = int(time_padding / 2) - 1
        else:
            time_padding = int(time_padding / 2)

        # Embedding and positional encoding
        self.embedding = nn.Sequential(
                            nn.Conv2d(1, n_maps,
                                      (1, position_kernel),
                                      stride=(1, 1),
                                      padding=(0, position_padding)),
                            nn.BatchNorm2d(n_maps),
                            nn.AdaptiveAvgPool2d(((channels_kernel,
                                                   new_seq_len))),
                            nn.Conv2d(n_maps, n_maps, (channels_kernel, 1),
                                      stride=(channels_stride, 1),
                                      groups=n_maps),
                            nn.BatchNorm2d(n_maps),
                            Mish(),
                            nn.Dropout(dropout),
                            nn.Conv2d(n_maps, emb_size, (1, time_kernel),
                                      stride=(1, time_stride),
                                      padding=(0, time_padding)),
                            nn.BatchNorm2d(emb_size),
                            Mish(),
                            nn.Dropout(dropout),
                            Rearrange('b o c t -> b (c t) o')
                            )

    def forward(self,
                x: Tensor):

        """ Create embeddings with positional encoding.

        Args:
            x (tensor): Batch of trials of dimension
                        [batch_size x 1 x n_channels x seq_len].

        Returns:
            x (tensor): Batches of embeddings of dimension
                        [batch_size x new_seq_len x emb_size].
                        If padding, maintain seq_len value.
        """

        # Create embeddings with positional encoding
        x = self.embedding(x)
        return x


""" ********** Transformer Encoder ********** """


class TransformerEncoder(nn.Sequential):

    """ Multi-head attention inspired by:
        `"Attention Is All You Need"
        <https://arxiv.org/pdf/1606.08415v3.pdf>`_.
    """

    def __init__(self,
                 depth,
                 emb_size,
                 num_heads,
                 expansion,
                 dropout):

        """
        Args:
            depth (int): Number of Transformer layers.
            emb_size (int): Size of embedding vectors.
            num_heads (int): Number of heads in multi-head block.
            expansion (int): Expansion coefficient in FF block.
            forward_dropout (float): Dropout value.
        """

        super().__init__()
        dim = expansion * emb_size
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size,
                                                   nhead=num_heads,
                                                   dim_feedforward=dim,
                                                   dropout=dropout,
                                                   activation='gelu')
        norm = nn.LayerNorm(emb_size)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=depth,
                                             norm=norm)

        # Weight initialization
        self.encoder.apply(xavier_initialization)

    def forward(self,
                x: Tensor):

        """ Apply Transformer Encoder.

        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x seq_len x emb_size].

        Returns:
             out (tensor): Batch of trials with dimension
                           [batch_size x seq_len x emb_size].
        """
        x = rearrange(x, 'b s e -> s b e')
        out = self.encoder(x)
        out = rearrange(out, 's b e -> b s e')

        return out


""" ********** Gated Transformer Network ********** """


class GTN(nn.Module):

    """ Gated Transformer Network inspired by:
        `"Gated Transformer Networks for Multivariate
        Time Series Classification"
        <https://arxiv.org/pdf/2103.14438.pdf>`_.
        Implementation inspired by
        Predicts probability of spike occurence in a trial.

    Input (tensor): Batch of trials of dimension
                    [batch_size x n_channels x n_time_points].
    Output (tensor): Logits of dimension [batch_size x 1].
    """

    def __init__(self,
                 n_time_points=201,
                 channel_num_heads=1,
                 channel_dropout=0.1,
                 emb_size=30,
                 n_maps=5,
                 position_kernel=50,
                 channels_kernel=20,
                 channels_stride=1,
                 time_kernel=20,
                 time_stride=1,
                 positional_dropout=0.25,
                 depth=3,
                 num_heads=10,
                 expansion=4,
                 transformer_dropout=0.25):

        """
        Args:
            n_time_points (int): Number of time points in EEF/MEG trials.
            channel_num_heads (int): Number of heads in ChannelAttention.
            channel_dropout (float): Dropout value in ChannelAttention.
            emb_size (int): Size of embedding vectors in Temporal transforming.
            n_maps (int): Number of feature maps for positional encoding.
            position_kernel (int): Kernel size for positional encoding.
            channels_kernel (int): Kernel size for convolution on channels.
            channels_stride (int): Stride for convolution on channels.
            time_kernel (int): Kernel size for convolution on time axis.
            time_stride (int): Stride for convolution on channel axis.
            positional_dropout (float): Dropout value for positional encoding.
            depth (int): Depth of the Transformer encoder.
            num_heads (int): Number of heads in multi-attention layer.
            expansion (int): Expansion coefficient in Feed Forward layer.
            transformer_dropout (float): Dropout value after Transformer.
            n_windows (int): Number of time windows.
        """

        super().__init__()
        self.spatial_transforming = ChannelAttention(n_time_points,
                                                     channel_num_heads,
                                                     channel_dropout)
        self.embedding = PatchEmbedding(n_time_points, emb_size,
                                        n_maps, position_kernel,
                                        channels_kernel,
                                        channels_stride, time_kernel,
                                        time_stride, positional_dropout)
        self.encoder = TransformerEncoder(depth, emb_size,
                                          num_heads,
                                          expansion,
                                          transformer_dropout)
        flatten_size = emb_size * n_time_points
        self.classifier = nn.Sequential(nn.Linear(flatten_size, 1))

        # Weight initialization
        self.classifier.apply(normal_initialization)

    def forward(self,
                x: Tensor):

        """ Apply STT model.
        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x 1 x n_channels x n_time_points].

        Returns:
            x (tensor): Batch of trials with dimension
                        [batch_size x 1 x n_channels x n_time_points].
            attention_weights (tensor): Attention weights of channel attention.
            out (tensor): If n_windows == 1 --> Logits of dimension
                                               [batch_size].
        """

        # Spatial Transforming
        attention, attention_weights = self.spatial_transforming(x)

        # Embedding
        embedding = self.embedding(attention)

        # Temporal Transforming
        code = self.encoder(embedding)

        # Output
        out = self.classifier(code.flatten(1)).squeeze(1)

        return out, attention_weights


""" ********** Spatial Temporal Transformers ********** """


class STT(nn.Module):

    """ Spatial Temporal Transformer inspired by:
        `"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
        <https://arxiv.org/pdf/2106.11170.pdf>`_.
        Predicts probability of spike occurence in a trial.

    Input (tensor): Batch of trials of dimension
                    [batch_size x 1 x n_channels x n_time_points].
    Output (tensor): Logits of dimension [batch_size x 1].
    """

    def __init__(self, n_time_points=201,
                 channel_num_heads=1,
                 channel_dropout=0.1,
                 emb_size=30,
                 n_maps=5,
                 position_kernel=50,
                 channels_kernel=20,
                 channels_stride=1,
                 time_kernel=20,
                 time_stride=1,
                 positional_dropout=0.25,
                 depth=3,
                 num_heads=10,
                 expansion=4,
                 transformer_dropout=0.25):

        """
        Args:
            n_time_points (int): Number of time points in EEF/MEG trials.
            channel_num_heads (int): Number of heads in ChannelAttention.
            channel_dropout (float): Dropout value in ChannelAttention.
            emb_size (int): Size of embedding vectors in Temporal transforming.
            n_maps (int): Number of feature maps for positional encoding.
            position_kernel (int): Kernel size for positional encoding.
            channels_kernel (int): Kernel size for convolution on channels.
            channels_stride (int): Stride for convolution on channels.
            time_kernel (int): Kernel size for convolution on time axis.
            time_stride (int): Stride for convolution on channel axis.
            positional_dropout (float): Dropout value for positional encoding.
            depth (int): Depth of the Transformer encoder.
            num_heads (int): Number of heads in multi-attention layer.
            expansion (int): Expansion coefficient in Feed Forward layer.
            transformer_dropout (float): Dropout value after Transformer.
            n_windows (int): Number of time windows.
        """

        super().__init__()
        self.spatial_transforming = ChannelAttention(n_time_points,
                                                     channel_num_heads,
                                                     channel_dropout)
        self.embedding = PatchEmbedding(n_time_points, emb_size,
                                        n_maps, position_kernel,
                                        channels_kernel,
                                        channels_stride, time_kernel,
                                        time_stride, positional_dropout)
        self.encoder = TransformerEncoder(depth, emb_size,
                                          num_heads,
                                          expansion,
                                          transformer_dropout)
        flatten_size = emb_size * n_time_points
        self.classifier = nn.Sequential(nn.Linear(flatten_size, 1))

        # Weight initialization
        self.classifier.apply(normal_initialization)

    def forward(self,
                x: Tensor):

        """ Apply STT model.
        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x 1 x n_channels x n_time_points].

        Returns:
            x (tensor): Batch of trials with dimension
                        [batch_size x 1 x n_channels x n_time_points].
            attention_weights (tensor): Attention weights of channel attention.
            out (tensor): If n_windows == 1 --> Logits of dimension
                                               [batch_size].
        """

        # Spatial Transforming
        attention, attention_weights = self.spatial_transforming(x)

        # Embedding
        embedding = self.embedding(attention)

        # Temporal Transforming
        code = self.encoder(embedding)

        # Output
        out = self.classifier(code.flatten(1)).squeeze(1)

        return out, attention_weights


""" ********** RNN self-attention ********** """


class RNN_self_attention(nn.Module):

    """ RNN self-attention inspired by:
        `"Epileptic spike detection by recurrent neural
        networks with self-attention mechanism"
        <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9747560>`_.

    Input (tensor): Batch of trials of dimension
                    [batch_size x n_time_points x 1].
    Output (tensor): Logits of dimension [batch_size].
    """

    def __init__(self,
                 n_time_points,
                 input_size=1,):
        """
        Args:
            input_size (int): Input size (here: n_time_points).
        """

        super().__init__()
        self.input_size = input_size
        self.n_time_points = n_time_points
        self.LSTM_1 = nn.LSTM(input_size=input_size,
                              hidden_size=8,
                              num_layers=1,
                              batch_first=True)
        self.tanh = nn.Tanh()
        self.avgPool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.attention = nn.MultiheadAttention(num_heads=1, embed_dim=8)
        self.LSTM_2 = nn.LSTM(input_size=8, hidden_size=8, num_layers=1,
                              batch_first=True)
        self.classifier = nn.Linear(int(n_time_points/2), 1)

    def forward(self,
                x: Tensor):

        """ Apply 1D-RNN with self-attention model.
        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x n_time_points x 1].

        Returns:
            out (tensor): Logits of dimension [batch_size].
            attention_weights (tensor): Attention weights of channel attention.
        """

        # First LSTM
        self.LSTM_1.flatten_parameters()
        x, (_, _) = self.LSTM_1(x.transpose(1, 2))
        x = self.avgPool(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)

        # Self-attention Layer
        x_attention, attention_weights = self.attention(x, x, x)

        x = x + x_attention
        x = x.transpose(0, 1)

        # Second LSTM
        self.LSTM_2.flatten_parameters()
        x, (_, _) = self.LSTM_2(x)
        x = self.tanh(x)
        x = self.avgPool(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # Classifier
        out = self.classifier(x.flatten(1)).squeeze(1)

        return out, attention_weights
