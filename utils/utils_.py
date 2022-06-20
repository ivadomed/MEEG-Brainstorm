#!/usr/bin/env python

"""
This script gathers useful Python functions.

Usage: type "from utils import <function>" to use one of its functions.

Contributors: Ambroise Odonnat.
"""

import torch

import numpy as np

from loguru import logger
from torch.utils.data.sampler import WeightedRandomSampler


def define_device(gpu_id):

    """ Define the device used for the process of interest.

    Args:
        gpu_id (int): ID of the cuda device.

    Returns:
        cuda_available (bool): If True, cuda is available.
        device (str): Cuda device.
    """

    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available()
                          else "cpu")
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


def get_next_batch(id,
                   iter_loader,
                   loaders):

    """ Loop in a dataloader. """

    try:
        x, y = next(iter_loader[id])
    except StopIteration:
        iter_loader[id] = iter(loaders[id])
        x, y = next(iter_loader[id])

    return x, y


def get_spike_events(spike_time_points,
                     n_time_points):

    """ Compute array of dimension [n_time_points]
        with 1 when a spike occurs and 0 otherwise.

    Args:
        spike_time_points (array): Contains time points of spike events.
        n_time_points (int): Number of time points.

    Returns:
        spike_events (array): Binary array of dimension [n_time_points]
                              with 1 when a spike occurs and 0 otherwise.
    """

    spike_events = np.zeros(n_time_points)
    for time in spike_time_points:
        spike_events[time] = 1

    return spike_events.astype(int)


def get_spike_windows(spike_events,
                      n_windows):

    """
    Compute tensor of dimension [batch_size x n_time_windows]
    with 1 when a spike occurs in the time window and 0 otherwise.

    Args:
        spike_events (tensor): Tensor of dimension [batch_size x n_time_points]
                               whith 1 when a spike occurs and 0 otherwise.
        n_windows (int): Number of time windows.

    Return:
        spike_windows (tensor): Tensor of dimension
                                [batch_size x n_time_windows]
                                with 1 when a spike occurs
                                in the time window and 0 otherwise.
    """

    # Split spike_events in n_time_windows time windows
    spike_windows = []
    spike_events = np.array(spike_events)
    chunks = np.array_split(spike_events, n_windows, axis=-1)

    # Put 1 when a spike occurs in the time window, 0 otherwise
    for i, chunk in enumerate(chunks):
        is_spike = int((chunk.sum(axis=-1) > 0))
        spike_windows.append(is_spike)

    return np.asarray(spike_windows, dtype='int64')


def normal_initialization(m):

    """ Initialize model weight with normal. """
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def get_pos_weight(labels):

    """
    Compute weight for positive class.
    If no positive examples in the dataset, return 1.

    Args:
        labels (list): Labels in the training dataset recovered
                       with labels[subject][session][trial].

    Return:
        pos_weight (tensor): Positive class weight.
    """

    neg, pos = 0, 0
    for id in range(len(labels)):
        for n_sess in range(len(labels[id])):
            for n_trial in range(len(labels[id][n_sess])):
                label = labels[id][n_sess][n_trial]
                if label == 1:
                    pos += 1
                else:
                    neg += 1

    # Compute the corresponding weights
    if pos:
        pos_weight = torch.as_tensor(neg / pos)
    else:
        pos_weight = torch.ones(1)

    return pos_weight


def pad_tensor(x,
               n_pads,
               dim):

    """
    Pad up to n_pads with lowest int value a given tensor on dimension dim .
    Args:
        x (tensor): Tensor to pad.
        n_pads (int): Size to pad to.
        dim (int):  Dimension to pad on.

    Eeturn:
        A new tensor padded to n_pads on dimension dim.
    """

    pad_size = list(x.shape)
    pad_size[dim] = n_pads-x.shape[dim]

    return torch.cat([torch.Tensor(x), torch.zeros(*pad_size)], dim=dim)


def reset_weights(m):

    """
    Reset model weights to avoid weight leakage

    Args:
        m (nn.Module): Model to reset.
    """
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()


def weighted_sampler(labels):

    """ Create weighted sampler to tackle class imbalance.
        Oversample ied trials in dataloader.

    Args:
        labels (tensor): Labels.

    Returns:
        sampler (Sampler): Weighted sampler
    """

    class_count = torch.bincount(torch.tensor(labels))
    class_weighting = 1. / class_count
    weights = class_weighting[labels]
    num_samples = len(weights)
    sampler = WeightedRandomSampler(weights,
                                    num_samples,
                                    replacement=False)
    return sampler


def xavier_initialization(m):

    """ Initialize model weight with xavier uniform.
    Args:
        m (nn.Module): Model.
    """

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
