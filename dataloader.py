#!/opt/anaconda3/bin/python

"""
This script is used to create train and test datasets and dataloaders.

Usage: type "from get_dataloader import <function>" to use on of its functions.

Contributors: Ambroise Odonnat.
"""

import numpy as np 
import torch

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from utils import get_class_distribution


def train_test_dataset(allData, allLabels, train_size, shuffle, random_state):
    
    """
    Split into train and test dataset.
    
    Args:
        allData (array): Trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points),
        allLabels (array): Corresponding labels,
        train_size (float): Proportion of dataset taken for training,
        test_size (float): Proportion of dataset taken for testing,
        shuffle (bool): Shuffle dataset before the split,
        random_state (int): Insure reproductible shuffle.
        
    Returns:
        tuple: train_data (array): Trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points) for training,
               train_labels (array): Corresponding labels,
               test_data (array): Trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points) for testing,
               test_labels (array): Corresponding labels.
    """
    
    # Split train and test dataset
    train_data, test_data, train_labels, test_labels = train_test_split(allData, allLabels,\
                                                                        train_size = train_size,\
                                                                        shuffle = shuffle,\
                                                                        random_state = random_state)
    
    return train_data, train_labels, test_data, test_labels


def get_dataloader(data, labels, batch_size, num_workers, balanced):

    """
    Get dataloader.
    
    Args:
        data (array): Trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points),
        labels (array): Corresponding labels,
        batch_size (float): Size of batches,
        num_workers (float): Number of loader worker processes,
        balanced (bool): Weight classes to compensate unbalance between classes.
        
    Returns:
        tuple: train_data (array): Trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points) for training,
               train_labels (array): Corresponding labels,
               test_data (array): Trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points) for testing,
               test_labels (array): Corresponding labels.
    """
    
    # To correct unbalance between classes, during training for instance
    if balanced:
        
        # Recover weighted sampler
        count_label = get_class_distribution(data, labels, False)
        data, labels = torch.from_numpy(data),torch.from_numpy(labels)    
        class_count = [i for i in count_label.values()]
        class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
        class_weights_all = class_weights[labels]
        weighted_sampler = WeightedRandomSampler(weights=class_weights_all,num_samples=len(class_weights_all),replacement=True)
        
        # Get dataloader
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset = dataset, batch_size = batch_size,\
                                shuffle = False, num_workers = num_workers, sampler = weighted_sampler)
    
    # To shuffle trials during testing for instance
    else:
        
        # Get dataloader
        data, labels = torch.from_numpy(data),torch.from_numpy(labels)    
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset = dataset, batch_size = batch_size,\
                            shuffle = True, num_workers = num_workers)

    return dataloader
