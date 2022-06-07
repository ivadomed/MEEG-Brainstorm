#!/usr/bin/env python

"""
This script is used to apply the mix-up strategy proposed in
`"mixup: BEYOND EMPIRICAL RISK MINIMIZATION"
<https://arxiv.org/pdf/1710.09412.pdf>`_. Implementation inspired by:
`<https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py>`_.

Usage: type "from mix-up import <function>" to use one of its functions.

Contributor: Ambroise Odonnat.
"""

import torch
import numpy as np


def mixup_data(batch_x,
               batch_y,
               device,
               beta=0.2):

    """ Compute mixed inputs, shuffle labels and convex coefficient.

    Args:
        batch_x (tensor): Batch of data.
        batch_y (tensor): Batch of labels.
        device (device): Device.
        beta (float): Parameter of the Beta Law.

    Returns:
        mixed_batch_x (tensor): Batch of mixed data.
        shuffle_batch_y (tensor): Shuffled batch of labels.
        lambd (float): Coefficient of the convex combination.
    """

    # Define convex combination coefficient
    if beta > 0:
        lambd = np.random.beta(beta, beta)
    else:
        lambd = 1
    batch_size = batch_x.size()[0]
    index = torch.randperm(batch_size).to(device=device)

    # Compute mix data
    mixed_batch_x = lambd * batch_x + (1-lambd) * batch_x[index, :]
    shuffle_batch_y = batch_y[index]

    return mixed_batch_x, shuffle_batch_y, lambd


def mixup_criterion(criterion,
                    output,
                    batch_y,
                    shuffle_batch_y,
                    lambd):

    """ Compute convex combination of loss.

    Args:
        output (tensor): Output of model.
        batch_y (tensor): Batch of labels.
        shuffle_batch_y (tensor): Shuffled batch of labels.
        lambd (float): Coefficient of the convex combination.

    Returns:
        Convex combination of loss.
    """

    return (lambd * criterion(output, batch_y) + (1-lambd)
            * criterion(output, shuffle_batch_y))
