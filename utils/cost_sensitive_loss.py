#!/usr/bin/env python

"""
This script is used to implement custom losses inspired by:
`"Cost-Sensitive Regularization for Diabetic
Retinopathy Grading from Eye Fundus Images"
<https://arxiv.org/pdf/2010.00291.pdf>`_.

Usage: type "from custom_losses import <class>" to use one of its classes.

Contributor: Ambroise Odonnat.
"""

import torch

import numpy as np

from torch import nn


class CostSensitiveLoss(nn.Module):

    """ Implement a cost-sensitive loss inspired by:
        `"Cost-Sensitive Regularization for Diabetic
        Retinopathy Grading from Eye Fundus Images"
        <https://arxiv.org/pdf/2010.00291.pdf>`_.
    """

    def __init__(self,
                 criterion,
                 lambd):

        """
        Args:
            criterion: Criterion.
            lambd (float): Modulate influence of the
                           cost-sensitive regularizer.
        """

        super().__init__()

        self.criterion = criterion
        self.lambd = lambd

        # Compute cost-sensitive matrix
        M = np.array([[0, 0], [1, 0]])
        self.M = torch.from_numpy(M)

    def forward(self,
                logits,
                targets):

        """
        Args:
            pred (tensor): Batch of logits.
            targets (tensor): Batch of target.

        Return:
            loss (float): Mean loss value on the batch.
        """

        # Compute mean loss on the batch
        loss = self.criterion(logits, targets)

        # Recover prediction
        pred = 1 * (logits > 0.5)

        # Compute cost-sensitive regularization
        CS = self.M[pred.long(), targets.long()].float()
        balanced = CS.sum(axis=-1).mean()
        loss += self.lambd * balanced

        return loss


def get_criterion(criterion,
                  cost_sensitive,
                  lambd):

    """ Add cost-sensitive regularizer to the loss function.

    Args:
        criterion: Criterion.
        cost_sensitive (bool): If True, add cost-sensitive regularizer.
        lambd (float): Modulate influence of the cost-sensitive regularizer.

    Return:
        criterion: Criterion.
    """

    # Apply cost-sensitive method
    if cost_sensitive:
        return CostSensitiveLoss(criterion, lambd)
    else:
        return criterion
