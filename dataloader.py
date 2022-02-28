#!/opt/anaconda3/bin/python

"""
This script is used to create train and test datasets and dataloaders.

Usage: type "from model import <class>" to use one of its class.
       type "from model import <function>" to use one of its function.

Contributors: Ambroise Odonnat.
"""

import numpy as np 
import torch

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from utils import get_class_distribution


class Loader:
    
    def __init__(self, allData, allLabels, train_size,\
                 batch_size, num_workers, random_state, shuffle, balanced):
        
        """
        Args:
            allData (array): Trials after CSP algorithm (n_trials x Nr x n_sample_points),
            allLabels (array): Corresponding labels,
            train_size (float): Proportion of dataset taken for training,
            batch_size (float): Size of batches,
            num_workers (float): Number of loader worker processes,
            random_state (int): Insure reproductible shuffle,
            shuffle (bool): Shuffle dataset before the split,
            balanced (bool): Apply weight classes to compensate unbalance between classes.
        """
            
        self.allData = allData
        self.allLabels = allLabels
        self.train_size = train_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_state = random_state
        self.shuffle = shuffle
        self.balanced = balanced
        
        
    def train_test_dataset(self, allData, allLabels, train_size, shuffle, random_state):

        """
        Split into train and test dataset.

        Args:
            allData (array): Trials after CSP algorithm (n_trials x Nr x n_sample_points),
            allLabels (array): Corresponding labels,
            train_size (float): Proportion of dataset taken for training,
            shuffle (bool): Shuffle dataset before the split,
            random_state (int): Insure reproductible shuffle.

        Returns:
            tuple: train_data (array): Trials after CSP algorithm (n_trials x Nr x n_sample_points) for training,
                   train_labels (array): Corresponding labels,
                   test_data (array): Trials after CSP algorithm (n_trials x Nr x n_sample_points) for testing,
                   test_labels (array): Corresponding labels.
        """

        # Split train and test dataset
        train_data, test_data, train_labels, test_labels = train_test_split(allData, allLabels,\
                                                                            train_size = train_size,\
                                                                            shuffle = shuffle,\
                                                                            random_state = random_state)
    
        return train_data, train_labels, test_data, test_labels


    def get_dataloader(self, data, labels, batch_size, num_workers, balanced):

        """
        Get dataloader.

        Args:
            data (array): Trials after CSP algorithm with expanded dimension of size (n_trials x 1 x Nr x n_sample_points),
            labels (array): Corresponding labels,
            batch_size (float): Size of batches,
            num_workers (float): Number of loader worker processes,
            balanced (bool): Apply weight classes to compensate unbalance between classes.

        Returns:
            tuple: dataloader (Dataloader): Dataloader of trials batches of size (batch_size x 1 x n_channels x n_sample_points).
        """

        # To correct unbalance between classes, during training for instance
        if balanced:

            # Recover weighted sampler
            count_label = get_class_distribution(labels, False)
            class_count = [count_label[k] for k in np.unique(labels)]
            class_weights = 1./torch.tensor(class_count, dtype=torch.float)
            data, labels = torch.from_numpy(data),torch.from_numpy(labels)              
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
    
    
    def train_test_dataloader(self):
        
        """
        Returns:
            tuple: train_dataloader (Dataloader): Dataloader of trials batches of size (batch_size x 1 x n_channels x n_sample_points) for training,
                   test_dataloader (Dataloader): Dataloader of trials batches of size (batch_size x 1 x n_channels x n_sample_points) for testing.
        """
                
        # Split data and labels in train and test
        data_train, labels_train, data_test, labels_test = self.train_test_dataset(self.allData, self.allLabels,\
                                                                   self.train_size, self.shuffle, self.random_state)
        
        # Expand dimension to batch_size x 1 x n_channels x time points for convolution layers
        data_train = np.expand_dims(data_train, axis=1)
        data_test = np.expand_dims(data_test, axis = 1)

        # Recover train and test dataloader
        train_dataloader = self.get_dataloader(data_train, labels_train, self.batch_size, self.num_workers, self.balanced)
        test_dataloader = self.get_dataloader(data_test, labels_test, self.batch_size, self.num_workers, balanced = False)
        
        return train_dataloader, test_dataloader
        
