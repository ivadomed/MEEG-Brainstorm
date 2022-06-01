#!/usr/bin/env python

"""
This script is used to create dataloaders.

Usage: type "from model import <class>" to use class.
       type "from model import <function>" to use function.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import torch

from torch.utils.data import DataLoader, Dataset

from utils.utils_ import pad_tensor


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

    def pad_collate(self, batch):

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

    def __call__(self, batch):
        return self.pad_collate(batch)


class SingleChannelDataset(Dataset):

    def __init__(self, data, labels):

        """
        Args:
            data (array): Array of trials of dimension
                          [n_trials x n_time_points x 1].
            labels (array): Array of labels of dimension [n_trials].
        """
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def multi_channel_loader(data, labels, batch_size, shuffle, num_workers):

    """ Create dataloader for multi-channel trials.
        Input can be padded on the channel dimension
        with artifical zero channels to match highest
        number of channels in the batch.

    Args:
        data (list): List of array of trials of dimension
                     [n_trials x 1 x n_channels x n_time_points].
        labels (list): List of corresponding array of labels.
        batch_size (float): Batch size.
        shuffle (bool): If True, shuffle batches in dataloader.
        num_workers (float): Number of loader worker processes.

    Returns:
        dataloader (Dataloader): Dataloader of batches.
    """

    # Get dataloader
    dataset = []
    for id in range(len(data)):
        for n_trial in range(data[id].shape[0]):
            dataset.append((data[id][n_trial], labels[id][n_trial]))
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers,
                        collate_fn=PadCollate(dim=1))

    return loader


def single_channel_loader(data, labels, batch_size, shuffle, num_workers):

    """ Create dataloader for single-channel trials.
    Args:
        data (array): Array of trials of dimension
                        [n_trials x n_time_points x 1].
        labels (array): Array of labels of dimension [n_trials].
        batch_size (int): Batch size.
        shuffle (bool): If True, shuffle batches in dataloader.
        num_workers (int): Number of loader worker processed.

    Returns:
        loader (Dataloader): Dataloader of batches.
    """

    dataset = SingleChannelDataset(data, labels)
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)

    return loader
