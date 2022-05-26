#!/usr/bin/env python

"""
This script is used to implement custom losses inspired by:
`"Cost-Sensitive Regularization for Diabetic
Retinopathy Grading from Eye Fundus Images"
<https://arxiv.org/pdf/2010.00291.pdf>`_.

Usage: type "from custom_losses import <class>" to use one of its classes.

Contributors: Ambroise Odonnat.
"""

import torch

import numpy as np

from torch import nn
from torchmetrics.functional import f1_score, precision_recall


class CostSensitiveLoss(nn.Module):

    """ Implement a cost-sensitive loss inspired by:
        `"Cost-Sensitive Regularization for Diabetic
        Retinopathy Grading from Eye Fundus Images"
        <https://arxiv.org/pdf/2010.00291.pdf>`_.
    """

    def __init__(self, n_classes, lambd):

        """
        Args:
            n_classes (int): Number of classes in the dataset.
            lambd (float): Modulate the influence of
                           the cost-sensitive regularization.
        """

        super().__init__()

        self.criterion = nn.CrossEntropyLoss()
        self.n_classes = n_classes
        self.lambd = lambd

        # Compute cost-sensitive matrix
        M = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                M[i, j] = (j-i) ** 2

        self.M = torch.from_numpy(M)

    def forward(self, outputs, labels):

        """
        Args:
            outputs (tensor): Batches of logits of dimension
                              [batch_size x n_classes].
            labels (tensor): Bacthes of labels of dimension [batch_size].

        Return:
            loss (float): Mean loss value on the batch.
        """

        # Recover prediction
        prediction = torch.max(outputs.data, 1)[1]

        # Compute cost-sensitive regularization
        coeff = self.lambd

        # Modulate with the sensitivity and precision of the model
        precision, recall = precision_recall(prediction, labels,
                                             average='macro',
                                             num_classes=self.n_classes)
        if precision*recall:
            coeff = self.lambd * (-np.log(np.sqrt(recall*precision)))

        # Compute mean loss on the batch
        CS = self.M[prediction, labels]
        loss = self.criterion(outputs, labels)
        loss += coeff * CS.mean()

        return loss


class DetectionLoss(nn.Module):

    """ Implement a cost-sensitive loss inspired by:
        `"Cost-Sensitive Regularization for Diabetic
        Retinopathy Grading from Eye Fundus Images"
        <https://arxiv.org/pdf/2010.00291.pdf>`_.
    """

    def __init__(self, criterion, n_classes, lambd):

        """
        Args:
            criterion (Loss): Criterion.
            n_classes (int): Number of classes in the dataset.
            lambd (float): Modulate the influence of the
                           cost-sensitive regularization.
            lambd (float): Modulate the influence of
                           the cost-sensitive regularization.
        """

        super().__init__()

        self.criterion = criterion
        self.n_classes = n_classes
        self.lambd = lambd

        # Compute cost-sensitive matrix
        M = np.zeros((n_classes, n_classes))
        for i in range(n_classes):
            for j in range(i+1, n_classes):
                M[i, j] = (j-i) ** 2

        self.M = torch.from_numpy(M)

    def forward(self, outputs, labels):

        # Recover prediction
        prediction = (outputs > 0.5).int()

        # Compute mean loss on the batch
        loss = self.criterion(outputs, labels)

        # Compute cost-sensitive regularization
        CS = self.M[prediction.long(), labels.long()]
        balanced = CS.sum(axis=-1).mean()
        loss += self.lambd * balanced

        return loss


def get_classification_loss(n_classes, cost_sensitive, lambd):

    """ Build a custom cross-entropy loss with a cost-sensitive regularization.

    Args:
        n_classes (int): Number of classes in the dataset.
        cost_sensitive (bool): Build cost-sensitive cross entropy loss.
        lambd (float): Modulate the influence of the cost-sensitive weigh.

    Return:
        criterion (Loss): Criterion.
    """

    criterion = nn.CrossEntropyLoss()

    # Apply cost-sensitive method
    if cost_sensitive:
        return CostSensitiveLoss(n_classes, lambd)
    else:
        return criterion


def get_detection_loss(criterion, cost_sensitive, lambd):

    """
    Build a custom cross-entropy loss with a cost-sensitive regularization.

    Args:
        criterion (Loss): Criterion.
        cost_sensitive (bool): Build cost-sensitive cross entropy loss.
        lambd (float): Modulate the influence of the cost-sensitive weight,

    Return:
        criterion (Loss): Criterion.
    """

    # Apply cost-sensitive method
    if cost_sensitive:
        return DetectionLoss(criterion, 2, lambd)
    else:
        return criterion
