#!/usr/bin/env python

"""
This script is used to create dataloaders.

Usage: type "from model import <class>" to use class.
       type "from model import <function>" to use function.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import torch

import numpy as np

from torch.utils.data import DataLoader, random_split

from utils.utils_ import pad_tensor, weighted_sampler


class PadCollate():

    """ Custom collate_fn that pads according to the longest sequence in
        a batch of sequences.
    """

    def __init__(self, dim=1):

        """
        Args:
            dim (int): Dimension to pad on.
        """

        self.dim = dim

    def pad_collate(self,
                    batch):

        """
        Args:
            batch (list): List of (tensor, label).

        Return:
            xs (tensor): Padded data.
            ys (tensor): Labels.
        """

        # Find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))

        # Pad according to max_len
        data = map(lambda x: pad_tensor(x[0], n_pads=max_len, dim=self.dim),
                   batch)
        labels = map(lambda x: torch.tensor(x[1]), batch)

        # Stack all
        xs = torch.stack(list(data), dim=0)
        ys = torch.stack(list(labels), dim=0)

        return xs, ys

    def __call__(self,
                 batch):

        return self.pad_collate(batch)


class Loader():

    """ Create dataloader of data. """

    def __init__(self,
                 data,
                 labels,
                 balanced,
                 shuffle,
                 batch_size,
                 num_workers,
                 split_dataset=False,
                 seed=42):

        """
        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            balanced (bool): If True, number of trials with
                             spike events is limited.
            shuffle (bool): If True, shuffle batches in dataloader.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.
            split_dataset (bool): If True, split dataset into training,
                                  validation, test.
            seed (int): Seed for reproductibility.
        """

        self.data = data
        self.labels = labels
        self.balanced = balanced
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_dataset = split_dataset
        self.seed = seed

    def balance_pad_loader(self,
                           data,
                           labels,
                           batch_size,
                           num_workers):

        """ Create dataloader of data.
            Trials in a given batch have same number of channels
            using padding (add zero artificial channels).
            Oversample ied trials in the dataloader.

        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.

        Returns:
            dataloader (array): Array of subject specific dataloaders
                                ordered in decreasing order of data.
        """

        # Get dataloader
        dataset = []
        labels_list = []
        for id in range(len(data)):
            for n_sess in range(len(data[id])):
                for n_trial in range(len(data[id][n_sess])):
                    dataset.append((data[id][n_sess][n_trial],
                                    labels[id][n_sess][n_trial]))
                    labels_list.append(labels[id][n_sess][n_trial])
        sampler = weighted_sampler(labels_list)
        loader = DataLoader(dataset=dataset, batch_size=batch_size,
                            sampler=sampler, num_workers=num_workers,
                            collate_fn=PadCollate(dim=1))
        dataloader = [loader]

        return dataloader

    def pad_loader(self,
                   data,
                   labels,
                   shuffle,
                   batch_size,
                   num_workers):

        """ Create dataloader of data.
            Trials in a given batch have same number of channels
            using padding (add zero artificial channels).

        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            shuffle (bool): If True, shuffle batches in dataloader.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.

        Returns:
            dataloader (array): Array of dataloader.
        """

        # Get dataloader
        dataset = []
        for id in range(len(data)):
            for n_sess in range(len(data[id])):
                for n_trial in range(len(data[id][n_sess])):
                    dataset.append((data[id][n_sess][n_trial],
                                    labels[id][n_sess][n_trial]))

        loader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            collate_fn=PadCollate(dim=1))
        dataloader = [loader]

        return dataloader

    def train_val_test_dataloader(self,
                                  data,
                                  labels,
                                  shuffle,
                                  batch_size,
                                  num_workers,
                                  seed):

        """ Split dataset into training, validation, test dataloaders.
            Trials in a given batch have same number of channels
            using padding (add zero artificial channels).

        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            shuffle (bool): If True, shuffle batches in dataloader.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.
            seed (int): Seed for reproductibility.

        Returns:
            tuple: tuple of all dataloaders and the training labels.
        """

        # Get dataset of every tuple (data, label)
        dataset = []
        for id in range(len(data)):
            for n_sess in range(len(data[id])):
                for n_trial in range(len(data[id][n_sess])):
                    x = np.expand_dims(data[id][n_sess][n_trial], axis=0)
                    y = labels[id][n_sess][n_trial]
                    dataset.append((x, y))

        # Define training, validation, test splits
        N = len(dataset)
        ratio = 0.80
        train_size = int(ratio * N)
        test_size = N - train_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, test_dataset = random_split(dataset,
                                                   [train_size, test_size],
                                                   generator=generator)
        N = len(train_dataset)
        train_size = int(ratio * N)
        val_size = N - train_size
        train_dataset, val_dataset = random_split(train_dataset,
                                                  [train_size, val_size],
                                                  generator=generator)

        # Z-score normalization
        target_mean = np.mean([np.mean(data[0]) for data in train_dataset])
        target_std = np.mean([np.std(data[0]) for data in train_dataset])
        train_data, val_data, test_data = [], [], []
        train_labels = []
        for (x, y) in train_dataset:
            train_data.append(((x-target_mean) / target_std, y))
            train_labels.append(y)
        for (x, y) in val_dataset:
            val_data.append(((x-target_mean) / target_std, y))
        for (x, y) in test_dataset:
            test_data.append(((x-target_mean) / target_std, y))

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers,
                                  collate_fn=PadCollate(dim=1))
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                collate_fn=PadCollate(dim=1))
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=PadCollate(dim=1))

        return [train_loader], [val_loader], [test_loader], train_labels

    def load(self):
        if self.split_dataset:
            return self.train_val_test_dataloader(self.data,
                                                  self.labels,
                                                  self.shuffle,
                                                  self.batch_size,
                                                  self.num_workers,
                                                  self.seed)
        else:
            if self.balanced:
                return self.balance_pad_loader(self.data,
                                               self.labels,
                                               self.batch_size,
                                               self.num_workers)
            else:
                return self.pad_loader(self.data,
                                       self.labels,
                                       self.shuffle,
                                       self.batch_size,
                                       self.num_workers)
