from torch import nn
import torch


class fukumori2021RNN(nn.Module):
    def __init__(self, input_size, ):
        super().__init__()
        self.input_size = input_size
        self.LSTM_1 = nn.LSTM(input_size=input_size,
                              hidden_size=8,
                              num_layers=1,
                              batch_first=True)
        self.tanh = nn.Tanh()
        self.avgPool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.selfattention = nn.MultiheadAttention(num_heads=1, embed_dim=8)
        self.LSTM_2 = nn.LSTM(input_size=8, hidden_size=8, num_layers=1, batch_first=True)
        self.classifier = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # First LSTM
        x, (_, _) = self.LSTM_1(x)
        x = self.avgPool(x.transpose(1, 2))
        x = x.transpose(1, 2)
        x = x.transpose(0, 1)

        # Self-attention Layer
        x_attention, attention_weights = self.selfattention(x, x, x)

        x = x + x_attention
        x = x.transpose(0, 1)

        # Second LSTM
        x, (_, _) = self.LSTM_2(x)
        x = self.tanh(x)
        x = self.avgPool(x.transpose(1, 2))
        x = x.transpose(1, 2)

        # Classifier
        x = self.classifier(x.flatten(1))
        x = self.sigmoid(x)

        return x, attention_weights


#!/usr/bin/env python

"""
This script contains a model to detect spikes and a model
to count the number of spikes inspired by:
`"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
<https://arxiv.org/pdf/2106.11170.pdf>`_.

Usage: type "from models import <class>" to use one class.

Contributors: Ambroise Odonnat.
"""

import torch

from einops import rearrange
from einops.layers.torch import Reduce, Rearrange
from torch import nn
from torch import Tensor
from heads import Mish, RobertaClassifier, SpikeDetector


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

        """ Multi-head attention inspired by:
            `"Attention Is All You Need"
            <https://arxiv.org/pdf/1606.08415v3.pdf>`_.

        Args:
            emb_size (int): Size of embedding vectors (here: n_time_points).
                            Warning -> num_heads must be a
                                       dividor of emb_size !
            num_heads (int): Number of heads in multi-head block.
            dropout (float): Dropout value in multi-head block.
        """

        super().__init__()

        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor):

        """ Apply spatial transforming.
            Trials can be padded with zeros channels for same sequence length.

        Args:
            x (torch tensor): Batches of trials of dimension
                              [batch_size x 1 x n_channels x n_time_points].

        Returns:
            out (tensor): Batches of trials of dimension
                          [batch_size x 1 x n_channels x n_time_points].
        """

        x = torch.squeeze(x, dim=1)

        # padded channels should be ignored in self-attention
        key_padding_mask = (x.mean(dim=-1) == 0) & (x.std(dim=-1) == 0)

        # Multi-head attention
        x = rearrange(x, 'b s e -> s b e')
        out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask)
        out = rearrange(out, 's b e -> b s e')
        out = self.gamma * out.unsqueeze(1)

        return out


""" ********** Embedding and positional encoding ********** """


class PatchEmbedding(nn.Module):

    def __init__(self, seq_len, emb_size, n_maps, position_kernel,
                 position_stride, channels_kernel, channels_stride,
                 time_kernel, time_stride, dropout):

        """Positional encoding and embedding. Inspired by:
            `"EEGNet: a compact convolutional neural network for EEG-based
            brain–computer interfaces"
            <https://iopscience.iop.org/article/10.1088/1741-2552/aace8c/pdf>`_.

        Args:
            seq_len (int): Sequence length (here: n_time_points).
            emb_size (int): Size of embedding vectors.
            n_maps (int): Number of feature maps for positional encoding.
            position_kernel (int): Kernel size for positional encoding.
            position_stride (float): Stride for positional encoding.
            channels_kernel (int): Kernel size for convolution on channels.
            channels_stride (int): Stride for convolution on channels.
            time_kernel (int): Kernel size for convolution on time axis.
            time_stride (int): Stride for convolution on channel axis.
            dropout (float): Dropout value.
        """

        super().__init__()

        # Padding values to preserve seq_len
        position_padding = ((position_stride-1) * seq_len -
                            (position_stride+1) + position_kernel)
        position_padding = int(position_padding / 2)
        new_seq_len = (((seq_len + 1 - position_stride)
                        / position_stride) + 1)
        new_seq_len = int(new_seq_len / 2)
        time_padding = ((time_stride-1) * new_seq_len -
                        (time_stride+1) + time_kernel)
        time_padding = int(time_padding / 2) + 1

        # Embedding and positional encoding
        self.embedding = nn.Sequential(
                            nn.AdaptiveAvgPool2d(((channels_kernel,
                                                   seq_len))),
                            nn.Conv2d(1, n_maps, (1, position_kernel),
                                      stride=(1, position_stride),
                                      padding=(0, position_padding)),
                            nn.BatchNorm2d(n_maps),
                            nn.LeakyReLU(),
                            nn.Dropout(dropout),
                            nn.Conv2d(n_maps, n_maps, (channels_kernel, 1),
                                      stride=(channels_stride, 1),
                                      groups=n_maps),
                            nn.Conv2d(n_maps, emb_size, (1, time_kernel),
                                      stride=(1, time_stride),
                                      padding=(0, time_padding)),
                            nn.BatchNorm2d(emb_size),
                            nn.LeakyReLU(),
                            nn.Dropout(dropout),
                            Rearrange('b o c t -> b (c t) o')
                            )

    def forward(self, x: Tensor):

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


""" ********** Multi-head attention ********** """


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_size, num_heads, dropout):

        """ Multi-head attention inspired by:
            `"Attention Is All You Need"
            <https://arxiv.org/pdf/1606.08415v3.pdf>`_.

        Args:
            emb_size (int): Size of embedding vectors (here: n_time_points).
                            Warning -> num_heads must be a
                                       dividor of emb_size !
            num_heads (int): Number of heads in multi-head block.
            dropout (float): Dropout value in multi-head block.
        """

        super().__init__()

        self.attention = nn.MultiheadAttention(emb_size, num_heads, dropout)

    def forward(self, x: Tensor):

        """ Apply multi-head attention.

        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x seq_len x emb_size].

        Returns:
             out: Batch of trials with dimension
                  [batch_size x seq_len x emb_size].
        """
        x = rearrange(x, 'b s e -> s b e')
        out, _ = self.attention(x, x, x)
        out = rearrange(out, 's b e -> b s e')

        return out


""" ********** Feed-forward block ********** """


class FeedForwardBlock(nn.Sequential):

    def __init__(self, emb_size, expansion, dropout):

        """ Inspired by: `"Attention Is All You Need"
            <https://arxiv.org/pdf/1606.08415v3.pdf>`_.
            ! Novel method: Feed Forward with Mish activation
                            instead of GELU activation.

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

        """ Inspired by: `"Generating Long Sequences with Sparse Transformers"
            <https://arxiv.org/pdf/1904.10509.pdf>`_.

        Args:
            emb_size (int): Size of embedding vectors.
            num_heads (int): Number of heads in multi-head block.
            expansion (int): Expansion coefficient in FF block.
            dropout (float): Dropout value.
        """

        super().__init__(ResidualAdd(
                            nn.Sequential(
                                nn.LayerNorm(emb_size),
                                MultiHeadAttention(emb_size,
                                                   num_heads,
                                                   dropout),
                                nn.Dropout(dropout)
                                )
                            ),  # MHA block

                         ResidualAdd(
                            nn.Sequential(nn.LayerNorm(emb_size),
                                          FeedForwardBlock(emb_size,
                                                           expansion,
                                                           dropout),
                                          nn.Dropout(dropout)
                                          )
                            )  # Novel Feed forward block
                         )


class TransformerEncoder(nn.Sequential):

    def __init__(self, depth, emb_size, num_heads, expansion, dropout):

        """
        Args:
            depth (int): Number of Transformer layers.
            emb_size (int): Size of embedding vectors.
            num_heads (int): Number of heads in multi-head block.
            expansion (int): Expansion coefficient in FF block.
            forward_dropout (float): Dropout value.
        """

        super().__init__(*[TransformerEncoderBlock(emb_size, num_heads,
                                                   expansion, dropout)
                           for _ in range(depth)])


""" ********** Classification Model ********** """


class ClassificationBertMEEG(nn.Sequential):

    """ Determine the number of spikes in an EEG/MEG trial. Inspired by:
        `"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
        <https://arxiv.org/pdf/2106.11170.pdf>`_.

    Input (tensor): Batch of trials of dimension
                    [batch_size x 1 x n_channels x n_time_points].
    Output (tensor): Tensor of logits of dimension
                     [batch_size x n_classes].
    """

    def __init__(self, n_classes, n_time_points, attention_num_heads,
                 attention_dropout, spatial_dropout, emb_size, n_maps,
                 position_kernel, position_stride, channels_kernel,
                 channels_stride, time_kernel, time_stride, positional_dropout,
                 embedding_dropout, depth, num_heads, expansion,
                 transformer_dropout, classifier_dropout):

        """
        Args:
            n_classes (int): Number of classes in the dataset.
            n_channels (int): Number of channels in input trials.
            n_time_points (int): Number of time points in EEF/MEG trials.
            attention_num_heads (int): Number of heads in ChannelAttention.
            attention_dropout (float): Dropout value in ChannelAttention.
            spatial_dropout (float): Dropout value after Spatial transforming.
            emb_size (int): Size of embedding vectors in Temporal transforming.
            n_maps (int): Number of feature maps for positional encoding.
            position_kernel (int): Kernel size for positional encoding.
            position_stride (float): Stride for positional encoding.
            channels_kernel (int): Kernel size for convolution on channels.
            channels_stride (int): Stride for convolution on channels.
            time_kernel (int): Kernel size for convolution on time axis.
            time_stride (int): Stride for convolution on channel axis.
            positional_dropout (float): Dropout value for positional encoding.
            embedding_dropout (float): Dropout value after embedding block.
            depth (int): Depth of the Transformer encoder.
            num_heads (int): Number of heads in multi-attention layer.
                             ! Warning: num_heads must be a
                                        dividor of emb_size !
            expansion (int): Expansion coefficient in Feed Forward layer.
            transformer_dropout (float): Dropout value in Transformer.
            classifier_dropout (float): Dropout value in Classifier.
        """

        super().__init__(ResidualAdd(
                            nn.Sequential(
                                nn.LayerNorm(n_time_points),
                                ChannelAttention(n_time_points,
                                                 attention_num_heads,
                                                 attention_dropout),
                                nn.Dropout(spatial_dropout)
                                )
                            ),
                         # Spatial transforming

                         PatchEmbedding(n_time_points, emb_size,
                                        n_maps, position_kernel,
                                        position_stride, channels_kernel,
                                        channels_stride, time_kernel,
                                        time_stride, positional_dropout),
                         # Embedding and positional encoding

                         nn.Dropout(embedding_dropout),

                         TransformerEncoder(depth, emb_size, num_heads,
                                            expansion, transformer_dropout),
                         # Temporal transforming

                         RobertaClassifier(emb_size, n_classes,
                                           classifier_dropout)
                         # Classifier
                         )


""" ********** Classification Model ********** """


class DetectionBertMEEG(nn.Sequential):

    """ Detect spikes events times in an EEG/MEG trial. Inspired by:
        `"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
        <https://arxiv.org/pdf/2106.11170.pdf>`_.

    Input (tensor): Batch of trials of dimension
                    [batch_size x 1 x n_channels x n_time_points].
    Output (tensor): Tensor of logits of dimension
                     [batch_size x n_classes].
    """

    def __init__(self, n_time_points, attention_num_heads, attention_dropout,
                 spatial_dropout,  emb_size, n_maps, position_kernel,
                 position_stride, channels_kernel, channels_stride,
                 time_kernel, time_stride, positional_dropout,
                 embedding_dropout, depth, num_heads, expansion,
                 transformer_dropout, n_time_windows,
                 detector_dropout):

        """
        Args:
            n_time_points (int): Number of time points in EEF/MEG trials.
            attention_num_heads (int): Number of heads in ChannelAttention.
            attention_dropout (float): Dropout value in ChannelAttention.
            spatial_dropout (float): Dropout value after Spatial transforming.
            emb_size (int): Size of embedding vectors in Temporal transforming.
            n_maps (int): Number of feature maps for positional encoding.
            position_kernel (int): Kernel size for positional encoding.
            position_stride (float): Stride for positional encoding.
            channels_kernel (int): Kernel size for convolution on channels.
            channels_stride (int): Stride for convolution on channels.
            time_kernel (int): Kernel size for convolution on time axis.
            time_stride (int): Stride for convolution on channel axis.
            positional_dropout (float): Dropout value for positional encoding.
            embedding_dropout (float): Dropout value after embedding block.
            depth (int): Depth of the Transformer encoder.
            num_heads (int): Number of heads in multi-attention layer.
            expansion (int): Expansion coefficient in Feed Forward layer.
            transformer_dropout (float): Dropout value after Transformer.
            n_time_windows (int): Number of time windows.
            detector_dropout (float): Dropout value in spike detector block.
        """

        super().__init__(ResidualAdd(
                            nn.Sequential(
                                nn.LayerNorm(n_time_points),
                                ChannelAttention(n_time_points,
                                                 attention_num_heads,
                                                 attention_dropout),
                                nn.Dropout(spatial_dropout)
                                )
                            ),
                         # Spatial transforming,

                         PatchEmbedding(n_time_points, emb_size,
                                        n_maps, position_kernel,
                                        position_stride, channels_kernel,
                                        channels_stride, time_kernel,
                                        time_stride, positional_dropout),
                         # Embedding and positional encoding

                         nn.Dropout(embedding_dropout),

                         TransformerEncoder(depth, emb_size, num_heads,
                                            expansion, transformer_dropout),
                         # Temporal transforming

                         SpikeDetector(n_time_points, n_time_windows,
                                       emb_size, detector_dropout)
                         # Spike detector
                         )
