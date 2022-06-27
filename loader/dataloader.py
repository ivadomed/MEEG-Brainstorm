#!/usr/bin/env python

"""
This script is used to create dataloaders.

Usage: type "from model import <class>" to use class.
       type "from model import <function>" to use function.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import torch
import random

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
                 annotated_channels,
                 single_channels,
                 batch_size,
                 num_workers,
                 subject_LOPO=None,
                 seed=42):

        """
        Args:
            data (list): List of EEG trials in .edf format.
            labels (list): Corresponding labels.
            annotated_channels (list): channels where the spike occurs.
            single_channels (bool): If True, select just one channel in training.
            batch_size (int): Batch size.
            num_workers (int): Number of loader worker processes.
            split_dataset (bool): If True, split dataset into training,
                                  validation, test.
            seed (int): Seed for reproductibility.
        """

        self.data = data
        self.labels = labels
        self.annotated_channels = annotated_channels
        self.single_channels = single_channels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.subject_LOPO = subject_LOPO
        self.seed = seed

    def pad_loader(self,
                   data,
                   labels,
                   annotated_channels,
                   single_channels,
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
                    x = data[id][n_sess][n_trial]
                    y = labels[id][n_sess][n_trial]
                    chan = annotated_channels[id][n_sess]
                    if single_channels:
                        # Select only the channels where a spike occurs
                        if chan != []:
                            x = x[:, chan]
                        for i in range(x.shape[1]):
                            dataset.append((x[:, i], y))
                    else:
                        dataset.append((x, y))

        loader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers,
                            collate_fn=PadCollate(dim=1))

        return loader

    def LOPO_dataloader(self,
                        data,
                        labels,
                        annotated_channels,
                        single_channels,
                        batch_size,
                        num_workers,
                        subject_LOPO,
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
        # TODO add a seed
        # Get dataset of every tuple (data, label)
        subject_ids = np.asarray(list(data.keys()))

        train_subject_ids = np.delete(subject_ids,
                                np.where(subject_ids == subject_LOPO))
        size = int(0.20 * train_subject_ids.shape[0])


        if size > 1:
            val_subject_ids = np.asarray(random.sample(list(train_subject_ids),
                                                    size))
        else:
            val_subject_ids = np.asarray([np.random.choice(train_subject_ids)])
        for id in val_subject_ids:
            train_subject_ids = np.delete(train_subject_ids,
                                        np.where(train_subject_ids == id))
        print('Test on: {}, '
            'Validation on: {}'.format(subject_LOPO,
                                        val_subject_ids))

        # Training data
        train_data = []
        train_labels = []
        for id in train_subject_ids:
            train_data.append(data[id])
            train_labels.append(labels[id])

        # Z-score normalization
        target_mean = np.mean([np.mean([np.mean(data) for data in data_id])
                            for data_id in train_data])
        target_std = np.mean([np.mean([np.std(data) for data in data_id])
                            for data_id in train_data])
        train_data = [[np.expand_dims((data-target_mean) / target_std, axis=1)
                    for data in data_id] for data_id in train_data]

        # Validation data
        val_data = []
        val_labels = []
        for id in val_subject_ids:
            val_data.append(data[id])
            val_labels.append(labels[id])

        # Z-score normalization
        val_data = [[np.expand_dims((data-target_mean) / target_std,
                                    axis=1)
                    for data in data_id] for data_id in val_data]

        # Test data
        test_data = []
        test_labels = []
        test_data.append(data[subject_LOPO])
        test_labels.append(labels[subject_LOPO])

        # Z-score normalization
        test_data = [[np.expand_dims((data-target_mean) / target_std,
                                    axis=1)
                    for data in data_id] for data_id in test_data]

        train_loader = self.pad_loader(train_data,
                                       train_labels,
                                       annotated_channels,
                                       single_channels,
                                       shuffle=True,
                                       batch_size=batch_size,
                                       num_workers=num_workers)
        val_loader = self.pad_loader(val_data,
                                     val_labels,
                                     annotated_channels,
                                     single_channels=False,
                                     shuffle=False,
                                     batch_size=batch_size,
                                     num_workers=num_workers)
        test_loader = self.pad_loader(test_data,
                                      test_labels,
                                      annotated_channels,
                                      single_channels=False,
                                      shuffle=False,
                                      batch_size=batch_size,
                                      num_workers=num_workers)

        return train_loader, val_loader, test_loader, train_labels

    def train_val_test_dataloader(self,
                                data,
                                labels,
                                annotated_channels,
                                single_channels,
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
        dataset = []
        subject_ids = np.asarray(list(data.keys()))

        for id in subject_ids:
            for n_sess in range(len(data[id])):
                for n_trial in range(len(data[id][n_sess])):
                    x = np.expand_dims(data[id][n_sess][n_trial], axis=0)
                    y = labels[id][n_sess][n_trial]
                    chan = annotated_channels[id][n_sess]
                    dataset.append((x, y, chan))

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

        for (x, y, chan) in train_dataset:

            x = (x-target_mean) / target_std
            if single_channels:
                # Select only the channels where a spike occurs
                if chan != []:
                    x = x[:, chan]
                for i in range(x.shape[1]):
                    train_data.append((x[:, i], y))
            else:
                train_data.append((x, y))

            train_labels.append(y)
        for (x, y, _) in val_dataset:
            val_data.append(((x-target_mean) / target_std, y))
        for (x, y, _) in test_dataset:
            test_data.append(((x-target_mean) / target_std, y))

        train_loader = DataLoader(dataset=train_data, batch_size=batch_size,
                                    shuffle=True, num_workers=num_workers,
                                    collate_fn=PadCollate(dim=1))
        val_loader = DataLoader(dataset=val_data, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers,
                                collate_fn=PadCollate(dim=1))
        test_loader = DataLoader(dataset=test_data, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers,
                                    collate_fn=PadCollate(dim=1))

        return train_loader, val_loader, test_loader, train_labels

    def load(self):
        if self.subject_LOPO:
            return self.LOPO_dataloader(self.data,
                                        self.labels,
                                        self.annotated_channels,
                                        self.single_channels,
                                        self.batch_size,
                                        self.num_workers,
                                        self.subject_LOPO,
                                        self.seed)
        else:
            return self.train_val_test_dataloader(self.data,
                                                  self.labels,
                                                  self.annotated_channels,
                                                  self.single_channels,
                                                  self.batch_size,
                                                  self.num_workers,
                                                  self.seed)
