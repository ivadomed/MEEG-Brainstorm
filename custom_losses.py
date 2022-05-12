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

    """
    Implement a cost-sensitive loss inspired by:
    `"Cost-Sensitive Regularization for Diabetic
    Retinopathy Grading from Eye Fundus Images"
    <https://arxiv.org/pdf/2010.00291.pdf>`_.
    """

    def __init__(self, criterion, n_classes, lambd):

        """
        Args:
            criterion (Loss): Criterion.
            n_classes (int): Number of classes in the dataset.
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

    """
    Implement a cost-sensitive loss inspired by:
    `"Cost-Sensitive Regularization for Diabetic
    Retinopathy Grading from Eye Fundus Images"
    <https://arxiv.org/pdf/2010.00291.pdf>`_.
    """

    def __init__(self, criterion, n_classes, lambd):

        """
        Args:
            criterion (Loss): Criterion.
            n_classes (int): Number of classes in the dataset.
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

        """
        Args:
            outputs (tensor): Batches of logits of dimension
                              [batch_size x seq_len x 2].
            labels (tensor): Bacthes of labels of dimension
                             [batch_size x seq_len].

        Return:
            loss (float): Mean loss value on the batch.
        """

        # Recover prediction
        prediction = torch.max(outputs.data, 1)[1]

        # Compute cost-sensitive regularization
        coeff = self.lambd

        # Modulate with the sensitivity and precision of the model
        confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        for t, p in zip(labels.reshape(-1), prediction.reshape(-1)):
            confusion_matrix[t.long(), p.long()] += 1
        TP = confusion_matrix[1][1]
        TN = confusion_matrix[0][0]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]
        if FN*TP*FP*TN:
            coeff = - np.log(np.sqrt((TP/(TP+FN)) * (TP/(TP+FP))))
            coeff *= self.lambd

        # Compute mean loss on the batch
        CS = self.M[prediction, labels]
        loss = self.criterion(outputs, labels)
        loss += coeff * CS.mean()

        return loss


def get_classification_loss(n_classes, cost_sensitive, lambd):

    """
    Build a custom cross-entropy loss with a cost-sensitive regularization.

    Args:
        criterion (Loss): Criterion.
        n_classes (int): Number of classes in the dataset.
        cost_sensitive (bool): Build cost-sensitive cross entropy loss.
        lambd (float): Modulate the influence of the cost-sensitive weight,

    Return:
        criterion (Loss): Criterion.
    """

    criterion = torch.nn.CrossEntropyLoss()

    # Apply cost-sensitive method
    if cost_sensitive:
        return CostSensitiveLoss(criterion, n_classes, lambd)
    else:
        return criterion


def get_detection_loss(cost_sensitive, lambd):

    """
    Build a custom cross-entropy loss with a cost-sensitive regularization.

    Args:
        criterion (Loss): Criterion.
        cost_sensitive (bool): Build cost-sensitive cross entropy loss.
        lambd (float): Modulate the influence of the cost-sensitive weight,

    Return:
        criterion (Loss): Criterion.
    """

    criterion = torch.nn.CrossEntropyLoss()

    # Apply cost-sensitive method
    if cost_sensitive:
        return DetectionLoss(criterion, 2, lambd)
    else:
        return criterion
