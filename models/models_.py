#!/usr/bin/env python

"""
This script contains a model to detect spikes and a model
to count the number of spikes inspired by:
`"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
<https://arxiv.org/pdf/2106.11170.pdf>`_.

Usage: type "from models import <class>" to use one class.

Contributors: Ambroise Odonnat.
"""

from re import X
import torch

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torch import Tensor
from heads import Mish, RobertaClassifier, SpikeDetector


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
            dropout (float): Dropout value.
        """

        super().__init__()

        self.attention = nn.MultiheadAttention(emb_size, num_heads,
                                               dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(emb_size)

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

        temp = torch.squeeze(x, dim=1)

        # padded channels are ignored in self-attention
        key_padding_mask = (temp.mean(dim=-1) == 0) & (temp.std(dim=-1) == 0)
        temp = rearrange(temp, 'b s e -> s b e')
        temp, _ = self.spatial_transforming(temp, temp, temp,
                                            key_padding_mask=key_padding_mask)
        temp = rearrange(temp, 's b e -> b s e')
        x_attention = self.dropout(temp).unsqueeze(1)
        out = self.norm(x + x_attention)

        return out


""" ********** Embedding and positional encoding ********** """


class PatchEmbedding(nn.Module):

    def __init__(self, seq_len, emb_size, n_maps, position_kernel,
                 channels_kernel, channels_stride,
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
                            nn.LeakyReLU(),
                            nn.Dropout(dropout),
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


""" ********** Transformer Encoder ********** """


class TransformerEncoder(nn.Sequential):

    """ Multi-head attention inspired by:
        `"Attention Is All You Need"
        <https://arxiv.org/pdf/1606.08415v3.pdf>`_.
    """

    def __init__(self, depth, emb_size, num_heads, expansion, dropout):

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
                                                   n_heads=num_heads,
                                                   dim_feedforward=dim,
                                                   dropout=dropout,
                                                   activation='gelu')
        norm = nn.LayerNorm(emb_size)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                             num_layers=depth,
                                             norm=norm)

    def forward(self, x: Tensor):

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


""" ********** Classification and Detection heads ********** """


class Mish(nn.Module):

    """ Activation function inspired by:
        `<https://www.bmvc2020-conference.com/assets/papers/0928.pdf>`.
    """

    def __init__(self):

        super().__init__()

    def forward(self, x):

        return x*torch.tanh(F.softplus(x))


class RobertaClassifier(nn.Sequential):

    def __init__(self, emb_size, n_classes, dropout):

        """ Model inspired by
            `<https://zablo.net/blog/post/custom-classifier-on-bert-model-guide-polemo2-sentiment-analysis/>`_

        Args:
            emb_size (int): Size of embedding vector.
            n_classes (int): Number of classes.
            dropout (float): Dropout value.
        """

        super().__init__()

        self.classifier = nn.Sequential(nn.Dropout(dropout),
                                        Reduce('b v o -> b o',
                                               reduction='mean'),
                                        nn.Linear(emb_size, emb_size),
                                        Mish(),
                                        nn.Dropout(dropout),
                                        nn.Linear(emb_size, n_classes)
                                        )

        # Weight initialization
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, x):

        """ Compute logits used to obtain probability vector on classes.

        Args:
            x (tensor): Batch of dimension
                        [batch_size x seq_len x emb_size].

        Returns:
            x : Batch of dimension
                [batch_size x seq_len x emb_size].
            out: Tensor of logits of dimension [batch_size x n_classes].
        """

        out = self.classifier(x)
        return x, out


class SpikeDetector(nn.Sequential):

    def __init__(self, seq_len, n_time_windows, emb_size, dropout):

        """
        Args:
            seq_len (int): Sequence length (here: n_time_points).
            n_time_windows (int): Number of time windows.
            emb_size (int): Size of embedding vector.
            dropout (float): Dropout value.
        """

        super().__init__()

        self.predictor = nn.Sequential(nn.Dropout(dropout),
                                       Rearrange('b v o -> b o v'),
                                       nn.Linear(seq_len, n_time_windows),
                                       Rearrange('b o v -> b v o'),
                                       Mish(),
                                       nn.Dropout(dropout),
                                       nn.LayerNorm(emb_size),
                                       nn.Linear(emb_size, 1),
                                       Rearrange('b v o -> b (v o)')
                                       )

        # Weight initialization
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.02)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, x):

        """
        Predicts probability of spike in each time window w.

        Args:
            x (tensor): Batch of dimension
                        [batch_size x seq_len x emb_size].

        Returns:
            x : Batch of dimension
                [batch_size x seq_len x emb_size].
            out: Array of logits of dimension
                 [batch_size x n_time_windows].
        """

        out = self.predictor(x)
        return x, out


""" ********** Spatial Temporal Transformers ********** """


class STT(nn.Sequential):

    """ Spatial Temporal Transformer inspired by:
        `"Transformer-based Spatial-Temporal Feature Learning for EEG Decoding"
        <https://arxiv.org/pdf/2106.11170.pdf>`_.
        If task == 'classification', indicates if a trial contains spikes.
        If task == 'detection', indicates in which time windows
        spikes occur, if any.

    Input (tensor): Batch of trials of dimension
                    [batch_size x 1 x n_channels x n_time_points].
    Output (tensor): If task == 'detection' --> Logits of dimension
                                                [batch_size x n_time_windows].
                     If task == 'classification' --> Logits of dimension
                                                     [batch_size].
    """

    def __init__(self, n_time_points, channel_num_heads, channel_dropout,
                 emb_size, n_maps, position_kernel,
                 channels_kernel, channels_stride,
                 time_kernel, time_stride, positional_dropout,
                 depth, num_heads, expansion,
                 transformer_dropout, n_windows, task):

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
            task (str): Indicates whether the model detects epileptic spikes
                        events in a trial or indicates if spikes
                        occur in a trial.
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
        self.temporal_transforming = TransformerEncoder(depth, emb_size,
                                                        num_heads,
                                                        expansion,
                                                        transformer_dropout)
        self.detection_head = SpikeDetector(n_time_points,
                                            n_windows,
                                            emb_size)
        self.classification_head = RobertaClassifier(emb_size)

        # Define task
        error = "Incorrect task assigned."
        assert (task == 'detection') or (task == 'classification'), error
        self.task = task

    def forward(self, x: Tensor):

        """ Apply STT model.
        Args:
            x (tensor): Batch of trials with dimension
                        [batch_size x 1 x seq_len x emb_size].

        Returns:
             out (tensor): If task == 'detection' --> Logits of dimension
                                                      [batch_size x n_windows].
                           If task == 'classification' --> Logits of dimension
                                                           [batch_size].
        """

        # Spatial transforming
        # padded channels should be ignored in self-attention


        # Embedding
        x = self.embedding(x)

        # Temporal transforming
        x = self.temporal_transforming(x)

        # Detection
        if self.task == 'detection':
            x = self.detection_head(x)

        # Classification
        elif self.task == 'classification':
            x = self.classification_head(x)

        return x


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
                 channels_kernel, channels_stride,
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
                                        channels_kernel,
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
