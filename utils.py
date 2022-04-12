#!/usr/bin/env python

"""
This script gathers small Python functions.

Usage: type "from utils import <function>" to use on of its functions.

Contributors: Ambroise Odonnat.
"""

import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from loguru import logger
from torch.autograd import Variable


def get_spike_events(spike_time_points, n_time_points, freq):

    """
    Compute array of dimension [n_time_points] with 1 when a spike occurs and 0 otherwise.
    
    Args:
        spike_time_points (array): Contains time points when a spike occurs if any.
        n_time_points (int): Number of time points.
        freq (int): Sample frequence of the EEG/MEG signals.
        
    Returns:
        spike_events (array): Array of dimension [n_time_points] containing 1 when a spike occurs and 0 otherwise. 
    """
    
    spike_events = np.zeros(n_time_points)
    for time in spike_time_points:
        index = int(freq*time)
        spike_events[index] = 1
        
    return spike_events.astype(int)


def get_spike_windows(spike_events, n_time_windows):
    
    """
    Compute tensor of dimension [batch_size x n_time_windows] with 1 when a spike occurs in the time window and 0 otherwise.
    
    Args:
        spike_events (tensor): Tensor of dimension [batch_size x n_time_points] 
                               containing 1 when a spike occurs and 0 otherwise. 
        n_time_windows (int): Number of time windows.
        
    Return:
        spike_windows (tensor): Tensor of dimension [batch_size x n_time_windows] 
                                with 1 when a spike occurs in the time window and 0 otherwise
    """
    
    # Split spike_events in n_time_windows time windows
    batch_size = spike_events.size(0)
    spike_windows = np.zeros((n_time_windows,batch_size))
    chunks = torch.chunk(spike_events, n_time_windows, dim=-1)
    
    # Put 1 when a spike occurs in the time window, 0 otherwise
    for i,chunk in enumerate(chunks):
        is_spike = (chunk.sum(dim=-1) > 0).int()
        spike_windows[i] = is_spike
    spike_windows = torch.Tensor(spike_windows).t()
    
    return spike_windows


def get_class_distribution(labels, display=False, save=False, title=None):

    """
    Plot the distribution of labels.
    
    Args:
        labels (array): Labels in the dataset.
        display (bool): Display histogram of class repartition.
        save (bool): Save image.
        title (str): Name and format of saved file.
        
    Returns:
        count_labels (dictionnary): Keys are labels and values are the corresponding proportions.
    """
    
    count_labels = {k:0 for k in labels[::-1] }
    for k in labels:
        count_labels[k] += 1
        
    # Force dictionnary to be ordered by keys
    count_labels = dict(sorted(count_labels.items(), key=lambda item: item[0]))
    
    # Plot distribution
    if display:
        print("Class Distribution: \n",count_labels)
        sns.barplot(data = pd.DataFrame.from_dict([count_labels]).melt(),
                    x="variable", y="value").set_title('Class Distribution')
    if save:
        plt.savefig(title) 
        
    return count_labels


def define_device(gpu_id):
    
    """
    Define the device used for the process of interest.
    Args:
        gpu_id (int): ID of the cuda device.
        
    Returns:
        cuda_available (bool): If True, cuda is available.
        device (str): Cuda device.
    """
    
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    
    # Check cuda availability
    if cuda_available:
        
        # Set the GPU
        gpu_id = int(gpu_id)
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using GPU ID {gpu_id}")
    else:
        logger.info("Cuda is not available.")
        logger.info("Working on {}.".format(device))
        
    return cuda_available, device


def l1_regularization(w):
    
    """
    Compute L1 regularization penalty term.
    
    Args: 
        w: Weight of the model.
    
    Return: 
        Sum of absolute values of w. 
    """
    
    return torch.abs(w).sum()


def inverse_number_samples(labels):
    
    """
    Compute Inverse Number of Samples (INS).
    
    Args: 
        labels (array): Labels in the training dataset.
    
    Return: 
        class_weights (tensor): Inverse number of samples for each class. 
    """
        
    # Recover the number of samples for each class in the training dataset
    count_label = get_class_distribution(labels)
    class_count = [count_label[k] for k in np.unique(labels)]
    
    # Compute the corresponding weights
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    
    return class_weights


def inverse_square_root_number_samples(labels):
    
    """
    Compute Inverse Powered Number of Samples (IPNS).
    
    Args: 
        labels (array): Labels in the training dataset.
    
    Return: 
        class_weights (tensor): Inverse powered number of samples for each class. 
    """
    
    # Recover the number of samples for each class in the training dataset
    count_label = get_class_distribution(labels)
    class_count = [count_label[k] for k in np.unique(labels)]
    
    # Compute the corresponding weights
    class_weights = 1./np.power(torch.tensor(class_count, dtype = torch.float), 1/2) 
    
    return class_weights


def effective_number_samples(labels, beta=0.9):

    """
    Compute Effective Number of Samples (ENS) inspired by:
    "Class-Balanced Loss Based on Effective Number of Samples" `<https://arxiv.org/pdf/1901.05555.pdf>`_.
    
    Args: 
        labels (array): Labels in the training dataset.
        beta (float): Hyperparameters in [0,1[. Suggested values: 0.9,0.99,0.999,0.9999.

    Return: 
        class_weights (tensor): Inverse powered number of samples for each class. 
    """
    
    assert beta < 1, "Beta takes its values between 0 and 1 excluded, please modify the value of beta."
    
    # Recover the number of samples for each class in the training dataset
    count_labels = get_class_distribution(labels)
    class_count = [count_labels[k] for k in np.unique(labels)]
    
    # Compute the corresponding weights
    class_weights = (1-beta) / (1-np.power(beta, torch.tensor(class_count, dtype=torch.float))) 
    
    return class_weights

