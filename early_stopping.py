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

    def __init__(self, patience=10, verbose=False,
                 delta=0, path='checkpoint.pt'):

        """
        Args:
            patience (int): Number of epochs without validation
                            improvement before stopping training.
            verbose (bool): If True, prints a message for
                            each validation loss improvement.
            delta (float): Minimum change in the monitored quantity
                           to qualify as an improvement,
            path (str): Path for the checkpoint to be saved to.
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def save_checkpoint(self, val_loss, model):

        """
        Save model's parameters when validation loss decreases.

        Args:
            val_loss (float): current value of validation loss,
            model: Model used.
        """

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f}'
                  '--> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def __call__(self, val_loss, model):

        """
        Args:
            val_loss (float): Current value of validation loss.
            model: Model used.
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
