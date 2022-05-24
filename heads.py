#!/usr/bin/env python

"""
This script contains classification and spike detection heads
coming on top of the transformer encoder.
RobertaClassifier inspired by:
`<https://zablo.net/blog/post/custom-classifier-on-bert-model-guide-polemo2-sentiment-analysis/>`_.

Usage: type "from heads import <class>" to use one of its classes.

Contributors: Ambroise Odonnat.
"""

import torch
import torch.nn.functional as F

from einops.layers.torch import Rearrange, Reduce
from torch import nn


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

        self.classifier = nn.Sequential(
                             nn.Dropout(dropout),
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

        self.predictor = nn.Sequential(
                            nn.Dropout(dropout),
                            Rearrange('b v o -> b o v'),
                            nn.Linear(seq_len, n_time_windows),
                            Rearrange('b o v -> b v o'),
                            Mish(),
                            nn.Dropout(dropout),
                            nn.LayerNorm(emb_size),
                            Reduce('b v o -> b v', reduction='mean'),
                            nn.Sigmoid()
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
