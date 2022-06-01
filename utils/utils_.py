#!/usr/bin/env python

"""
This script gathers useful Python functions.

Usage: type "from utils import <function>" to use one of its functions.

Contributors: Ambroise Odonnat.
"""

import torch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from loguru import logger


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


def get_class_distribution(labels,
                           display=False,
                           save=False,
                           title=None):

    """ Plot the distribution of labels.

    Args:
        labels (array): Labels in the dataset.
        display (bool): Display histogram of class repartition.
        save (bool): Save image.
        title (str): Name and format of saved file.

    Returns:
        count_labels (dictionnary): Keys - >labels; values ->  proportions.
    """

    count_labels = {k: 0 for k in labels[::-1]}
    for k in labels:
        count_labels[k] += 1

    # Force dictionnary to be ordered by keys
    count_labels = dict(sorted(count_labels.items(), key=lambda item: item[0]))

    # Plot distribution
    if display:
        print("Class Distribution: \n", count_labels)
        sns.barplot(data=pd.DataFrame.from_dict([count_labels]).melt(),
                    x="variable", y="value").set_title('Class Distribution')
    if save:
        plt.savefig(title)

    return count_labels


def get_spike_events(spike_time_points,
                     n_time_points,
                     freq):

    """ Compute array of dimension [n_time_points]
        with 1 when a spike occurs and 0 otherwise.

    Args:
        spike_time_points (array): Contains time points of spike events.
        n_time_points (int): Number of time points.
        freq (int): Sample frequence of the EEG/MEG signals.

    Returns:
        spike_events (array): Binary array of dimension [n_time_points]
                              with 1 when a spike occurs and 0 otherwise.
    """

    spike_events = np.zeros(n_time_points)
    for time in spike_time_points:
        index = int(freq*time)
        spike_events[index] = 1

    return spike_events.astype(int)


def get_spike_windows(spike_events,
                      n_time_windows):

    """
    Compute tensor of dimension [batch_size x n_time_windows]
    with 1 when a spike occurs in the time window and 0 otherwise.

    Args:
        spike_events (tensor): Tensor of dimension [batch_size x n_time_points]
                               whith 1 when a spike occurs and 0 otherwise.
        n_time_windows (int): Number of time windows.

    Return:
        spike_windows (tensor): Tensor of dimension
                                [batch_size x n_time_windows]
                                with 1 when a spike occurs
                                in the time window and 0 otherwise.
    """

    # Split spike_events in n_time_windows time windows
    batch_size = spike_events.shape[0]
    spike_windows = np.zeros((n_time_windows, batch_size))
    chunks = torch.chunk(spike_events, n_time_windows, dim=-1)

    # Put 1 when a spike occurs in the time window, 0 otherwise
    for i, chunk in enumerate(chunks):
        is_spike = (chunk.sum(dim=-1) > 0).int()
        spike_windows[i] = is_spike
    spike_windows = torch.Tensor(spike_windows).t()

    return spike_windows


def normal_initialization(m):

    """ Initialize model weight with normal. """
    if isinstance(m, torch.nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.02)
        if m.bias is not None:
            m.bias.data.zero_()


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


def xavier_initialization(m):

    """ Initialize model weight with xavier uniform.
    Args:
        m (nn.Module): Model.
    """

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
