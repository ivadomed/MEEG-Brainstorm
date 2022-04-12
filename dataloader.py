#!/usr/bin/env python

"""
This script is used to create dataloaders.

Usage: type "from model import <class>" to use one of its classes.
       type "from model import <function>" to use one of its functions.
       
Contributors: Ambroise Odonnat.
"""


import torch

import numpy as np 

from torch.utils.data import DataLoader, TensorDataset


def get_dataloader(data, labels, batch_size, num_workers):

    """
    Get dataloader.
    
    Args:
        data (array): Trials after CSP algorithm of dimension [n_trials x 1 x n_channels x n_time_points].
        labels (array): Corresponding labels.
        batch_size (float): Size of batches.
        num_workers (float): Number of loader worker processes.
        
    Returns:
        dataloader (Dataloader): Dataloader of trials batches of dimension [batch_size x 1 x n_channels x n_time_points].
    """

    # Get dataloader
    data, labels = torch.from_numpy(data),torch.from_numpy(labels)    
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader
    

    
        
