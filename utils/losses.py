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
                 lambd=1e-3):

        """
        Args:
            lambd (float): Modulate influence of the
                           cost-sensitive regularizer.
        """

        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss()
        self.lambd = lambd
        self.sigmoid = nn.Sigmoid()

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
        p = self.sigmoid(logits)
        preds = 1 * (p > 0.5)

        # Compute cost-sensitive regularization
        CS = self.M[targets.long(), preds.long()].float()
        balanced = CS.sum(axis=-1).mean()
        loss += self.lambd * balanced

        return loss


class FocalLoss(nn.Module):

    """ Implement a focal loss inspired by:
        `"Focal Loss for Dense Object Detection"
        <https://arxiv.org/pdf/1708.02002.pdf>`_.
    """

    def __init__(self,
                 alpha=0,
                 gamma=2):

        """
        Args:
            alpha (float).
            gamma (float). 
        """

        super().__init__()

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                logits,
                targets):

        """
        Args:
            logits (tensor): Batch of logits.
            targets (tensor): Batch of target.

        Return:
            loss (float): Mean loss value on the batch.
        """

        # Compute mean loss on the batch
        loss = self.criterion(logits, targets)

        # Recover probability
        p = self.sigmoid(logits)

        # Compute focal weight
        p_t = p * targets + (1-p) * (1-targets)
        loss *= (1-p_t) ** self.gamma
        if self.alpha > 0:
            alpha_t = self.alpha * targets + (1-self.alpha) * (1-targets)
            loss *= alpha_t
    
        # Mean reduction
        loss = loss.mean()

        return loss

    
def get_criterion(criterion,
                  cost_sensitive,
                  lambd,
                  focal,
                  alpha,
                  gamma):

    """ Get criterion.

    Args:
        criterion (Criterion): Binary Cross-Entropy with logits loss.
        cost_sensitive (bool): If True, add cost-sensitive regularizer.
        lambd (float): Modulate influence of the cost-sensitive regularizer.
        focal (bool): If True, use focal loss.
        alpha (float): Modulate influence of the cost-sensitive regularizer.
        gamma (float): Modulate influence of the cost-sensitive regularizer.

    Return:
        criterion: Criterion.
    """

    # Apply cost-sensitive method
    if cost_sensitive:
        return CostSensitiveLoss(lambd)
    elif focal:
        return FocalLoss(alpha, gamma)
    else:
        return criterion
