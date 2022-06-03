#!/usr/bin/env python

"""
This script is used to create dataloaders.

Usage: type "from model import <class>" to use class.
       type "from model import <function>" to use function.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import torch

import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler

from utils.utils_ import pad_tensor


# TODO a mettre dans utils ?
def class_imbalance_sampler(labels, n_sample):
    class_count = torch.bincount(labels.squeeze())
    class_weighting = 1. / class_count
    sample_weights = class_weighting[labels]
    if n_sample==0:
        n_sample=1
    sampler = WeightedRandomSampler(sample_weights,
                                    int(n_sample),
                                    replacement=False)
    return sampler


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


class SingleChannelDataset(Dataset):

    def __init__(self,
                 data,
                 labels):

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


def pad_loader(data,
               labels,
               batch_size,
               shuffle,
               num_workers):

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
    # for id in range(len(data)):
    #     for n_trial in range(data[id].shape[0]):
    #         dataset.append((data[id][n_trial], labels[id][n_trial]))
    for id in range(len(data)):
        for n_sess in range(len(data[id])):
            for n_trial in range(len(data[id][n_sess])):
                dataset.append((data[id][n_sess][n_trial], labels[id][n_sess][n_trial]))
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers,
                        collate_fn=PadCollate(dim=1))

    return [loader]


def balance_pad_loader(data,
                   labels,
                   batch_size,
                   shuffle,
                   num_workers):

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
    datasets = []
    n_ied_segments = []
    label_distributions = []
    for id in range(len(data)):
        label_distribution = np.concatenate(labels[id])
        label_distributions.append(label_distribution)

        n_ied_segment = np.sum(label_distribution == 1)
        n_ied_segments.append(n_ied_segment)

        dataset = []
        for n_sess in range(len(data[id])):
            for n_trial in range(len(data[id][n_sess])):
                dataset.append((data[id][n_sess][n_trial], labels[id][n_sess][n_trial]))
        datasets.append(dataset)

    # TODO mean or median or quartile ?
    n_ied_segments = np.array(n_ied_segments)
    n_ied_segments_mean = np.mean(n_ied_segments)
    n_ied_segments[n_ied_segments >= n_ied_segments_mean] = n_ied_segments_mean
    loaders = []
    for id in range(len(data)):

        sampler = class_imbalance_sampler(torch.tensor(label_distributions[id]),
                                          2*n_ied_segments[id])

        loaders.append(DataLoader(dataset=datasets[id],
                                  batch_size=batch_size,
                                  sampler=sampler,
                                  num_workers=num_workers,
                                  collate_fn=PadCollate(dim=1)))

    # Sort the loaders in the desending order

    argsort = np.argsort(n_ied_segments)
    loaders = np.array(loaders)[np.flipud(argsort)]

    return loaders

def load_loaders(data,
                 labels,
                 batch_size,
                 shuffle,
                 num_workers,
                 balance):

    if balance:
        return balance_pad_loader(data,
                              labels,
                              batch_size,
                              shuffle,
                              num_workers)
    else:
        return pad_loader(data,
                              labels,
                              batch_size,
                              shuffle,
                              num_workers)