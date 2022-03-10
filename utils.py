#!/opt/anaconda3/bin/python

"""
This script is used to gather small Python functions.

Usage: type "from utils import <function>" to use on of its functions.

Contributors: Ambroise Odonnat.
"""

import torch

import pandas as pd
import seaborn as sns

from loguru import logger

def get_class_distribution(label, display):

    """
    Get proportion of each class in the dataset data.
    
    Args:
        data (array): Training trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points),
        label (array): Corresponding labels,
        display (bool): Display histogram of class repartition.
        
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
    
    """Define the device used for the process of interest.
    Args:
        gpu_id (int): GPU ID.
    Returns:
        Bool, device: True if cuda is available.
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


def compute_l1_loss(w):
    
    """
    Apply L1 regularization.
    
    Args: w: weight of the model.
    """
    
    return torch.abs(w).sum()

def compute_l2_loss(w):
    
    """
    Apply L2 regularization.
    
    Args: w: weight of the model.
    """
    
    return torch.square(w).sum()