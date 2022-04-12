#!/usr/bin/env python

"""
This script contains the architecture of the model inspired by
`"Time2Vec: Learning a Vector Representation of Time" <https://arxiv.org/pdf/1907.05321.pdf>`_.

Usage: type "from Time2Vec import <class>" to use one of its classes.

Contributors: Ambroise Odonnat.
"""

""""""


import torch

import numpy as np

from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


class SineFunction(nn.Module):
    
    """
    Wrap the sinus function as a nn.Module.
    """
    
    def __init__(self):
        
        super().__init__()
        
    def forward(self,x):
        
        return torch.sin(x)

    
    
class Time2Vec(nn.Module):
    
    """
    Apply Time2Vec positional encoding inspired by 
    `"Time2Vec: Learning a Vector Representation of Time" <https://arxiv.org/pdf/1907.05321.pdf>`_.
    """

    def __init__(self, seq_len, emb_size, n_channels):
        
        """
        Args:
            seq_len (int): Length of sequence (corresponds to the number of time points in the EEG/MEG trial).
            emb_size (int): Size of embedding vectors.
            n_channels (int): Number of channels after CSP projection.
        """
        
        super().__init__()
        
        # Compute periodic feature
        self.periodic = nn.Sequential(Reduce('b t c -> b t', reduction='mean'),
                                      nn.Linear(seq_len, seq_len)
                                     )
        
        # Compute non-periodic feature
        self.non_periodic = nn.Sequential(Reduce('b t c -> b t', reduction='mean'),
                                          nn.Linear(seq_len, seq_len),
                                          SineFunction()
                                         )
        
        # Create embedding
        self.embedding = nn.Linear(n_channels+2, emb_size)
        
    def forward(self, x):
        
        """
        Apply Time2Vec on input x.
        
        Args: 
            x (tensor): batch of dimension [batch_size x 1 x n_channels x n_time_points].
        
        Return:
            tensor: batch of embeddings of dimension [batch_size x n_time_points x emb_size].
        """
        
        x = rearrange(x, 'b o c t -> b t (o c)')
        
        # Compute periodic and non periodic features
        periodic = self.periodic(x)
        periodic = torch.unsqueeze(periodic,-1)
        non_periodic = self.non_periodic(x)
        non_periodic = torch.unsqueeze(non_periodic,-1)
        positional_encoding = torch.cat([periodic, non_periodic],-1)
        
        # Create embedding
        embedding = torch.cat([x,positional_encoding],-1)
        embedding = self.embedding(embedding)
        
        return embedding