#!/opt/anaconda3/bin/python

"""
This script is used to gather small Python functions.

Usage: type "from utils import <function>" to use on of its functions.

Contributors: Ambroise Odonnat.
"""

import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch.autograd import Variable
from loguru import logger


def get_spike_events(spikeTimePoints, N, freq):

    """
    Compute array with binary values of size the number of time points with 1 when a spike occurs and 0 elsewhere.
    
    Args:
        spikeTimePoints (array): Array of spike times (if none, empty array),
        N (int): Number of time points,
        freq (int): Sample frequence of the EEG/MEG signals.
        
    Returns:
        spikeEvents (array): Array with dimension N with 1 when a spike occurs and 0 elsewhere.
    """
    
    spikeEvents = np.zeros(N)
    for time in spikeTimePoints:
        index = int(freq*time)
        spikeEvents[index] = 1
        
    return spikeEvents.astype(int)


def get_spike_windows(spikeEvents, n_windows):
    
    """
    Split a given array of dimension n_time_points into n_windows map each window on 0
    if no spike occurs in the given time window and 1 otherwize.
    
    Args:
        spikeEvents (tensor): Tensor ofdimension n_windows with value 1 if a spike occurs in the given window, 0 otherwise,
        n_window (int): Number of time windows.
        
    Return:
        tensor of dimension n_windows with value 1 if a spike occurs in the given window, 0 otherwise.
    """
    
    # Split spikeEvents in n_windows time windows
    batch_size = spikeEvents.size(0)
    spikeWindows = np.zeros((n_windows,batch_size))
    chunks = torch.chunk(spikeEvents,n_windows,dim = -1)
    for i,chunk in enumerate(chunks):
        l = (chunk.sum(dim = -1) > 0).int()
        spikeWindows[i] = l
    return torch.Tensor(spikeWindows).t()


def get_class_distribution(label, display = False, save = False, title = None):

    """
    Get proportion of each class in the dataset data.
    
    Args:
        data (array): Training trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points),
        label (array): Corresponding labels,
        display (bool): Display histogram of class repartition,
        save (bool): Save image,
        title (str): name and format of saved file.
        
    Returns:
            count_label (dictionnary): Keys are label and values are corresponding proportion.
    """
    
    count_label = {k:0 for k in label[::-1] }
    for k in label:
        count_label[k] += 1
        
    # Force dictionnary to be ordered by keys
    count_label = dict(sorted(count_label.items(), key=lambda item: item[0]))
    
    # Plot distribution
    if display:
        print("Class Distribution: \n",count_label)
        sns.barplot(data = pd.DataFrame.from_dict([count_label]).melt(), x = "variable", y="value")\
        .set_title('Class Distribution')
    
    if save:
        plt.savefig(title) 
        
    return count_label


def check_balance(dataloader, n_classes, n_epochs, display):
     
    """
    Plot histogramm of class distribution to check balance in the dataloader.
    
    Args:
        dataloader (Dataloader): Dataloader of training trials batches of size (batch_size x 1 x n_channels x n_sample_points),
        n_classes (int): Number of classes,
        display (bool): Display histogram of class repartition.
        
    Returns:
            acc (dictionnary): Keys are label and values are corresponding proportion.
    """
    
    acc = {k:0 for k in range(n_classes)}
    N = len(dataloader.dataset)
    for epoch in range(n_epochs):
        for i, (data, labels) in enumerate(dataloader):
            for k in acc.keys():
                acc[k] += int(torch.sum(labels == k)) 
                
    # Plot distribution
    if display:
        print("Class Distribution: \n",acc)
        sns.barplot(data = pd.DataFrame.from_dict([acc]).melt(), x = "variable", y="value",)\
        .set_title("Class Distribution")
    return acc


def define_device(gpu_id):
    
    """
    Define the device used for the process of interest.
    Args:
        gpu_id (int): ID of the cuda device.
                      Default: 0
    Returns:
        tuple : cuda_available (bool): True if cuda device is available.
                device (str): Cuda device.
    """
    
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    
    if not cuda_available:
        logger.info("Cuda is not available.")
        logger.info("Working on {}.".format(device))
    if cuda_available:
        
        # Set the GPU
        gpu_id = int(gpu_id)
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using GPU ID {gpu_id}")
        
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
    class_weights = 1. / torch.tensor(class_count, dtype = torch.float)
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
    class_weights = 1. / np.power(torch.tensor(class_count, dtype = torch.float), 1 / 2) 
    return class_weights


def effective_number_samples(labels, beta = 0.9):

    """
    Compute Effective Number of Samples (ENS) inspired by `<https://arxiv.org/pdf/1901.05555.pdf>`.
    As the number of samples of a given class increases, the contribution of each newly added data point diminishes.
    The idea is to measure a small neighboring region rather than a single data point.
    
    Args: 
        labels (array): Labels in the training dataset.
        beta (float): Hyperparameters in [0,1[ representing the contribution of each new sample to the effective volume of all samples in the training dataset.
                      Suggested values for beta: 0.9,0.99,0.999,0.9999.

    
    Return: 
        class_weights (tensor): Inverse powered number of samples for each class. 
    """
    
    assert beta < 1, "Beta takes its values between 0 and 1 excluded, please modify the value of beta."
    
    # Recover the number of samples for each class in the training dataset
    count_label = get_class_distribution(labels)
    class_count = [count_label[k] for k in np.unique(labels)]
    
    # Compute the corresponding weights
    class_weights = (1 - beta) / (1 - np.power(beta, torch.tensor(class_count, dtype = torch.float))) 
    return class_weights

