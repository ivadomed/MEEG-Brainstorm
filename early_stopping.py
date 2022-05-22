#!/usr/bin/env python

"""
This script is used to do early stopping during training. Inspired by:
`<https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py>`.

Usage: type "from early_stopping import <class>" to use one of its classes.

Contributors: Ambroise Odonnat.
"""

import torch

import numpy as np


class EarlyStopping():

    """
    Stop the training if validation loss doesn't improve after a given time.
    Inspired by:
    `<https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py>`.
    """

    def __init__(self, patience=10):

        """
        Args:
            patience (int): Number of epochs without validation
                            improvement before stopping training.
        """

        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):

        """
        Args:
            val_loss (float): Current value of validation loss.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss
        elif score <= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
