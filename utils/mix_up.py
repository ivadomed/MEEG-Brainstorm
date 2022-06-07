#!/usr/bin/env python

"""
This script is used to apply the mix-up strategy proposed in
`"mixup: BEYOND EMPIRICAL RISK MINIMIZATION"
<https://arxiv.org/pdf/1710.09412.pdf>`_. Implementation inspired by:
`<https://github.com/facebookresearch/mixup-cifar10/blob/main/train.py>`_.

Usage: type "from mix-up import <function>" to use one of its functions.

Contributor: Ambroise Odonnat.
"""

import numpy as np


def mixup_data(batch_x,
               batch_y,
               beta=1.0,
               use_cuda=True):

    """ Returns mixed inputs, pairs of targets, and lambda. """
    
    if beta > 0:
        lam = np.random.beta(beta, beta)
    else:
        lam = 1
    batch_size = batch_x.size()[0]
    device = batch_x.device
    index = torch.randperm(batch_size).to(device=device)
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)