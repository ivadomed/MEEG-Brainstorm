#!/usr/bin/env python

"""
This script is used to train and evaluate models.

Usage: type "from training import <class>" to use one of its classes.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import copy
from tkinter import N
import torch

import numpy as np

from sklearn.metrics import f1_score


class make_model():

    def __init__(self,
                 model,
                 loader_train,
                 loader_val,
                 optimizer,
                 criterion,
                 parameters,
                 n_epochs,
                 patience=None):

        """
        Args:
            model (nn.Module): Model.
            loader (Sampler): Generator of n_train EEG samples for training.
            loader_valid (Sampler): Generator of n_val samples for validation.
            optimizer (optimizer): Optimizer.
            criterion (Loss): Loss function.
            parameters : Parameters of the model.
            n_epochs (int): Maximum number of epochs to run.
            patience (int): Indicates how many epochs without improvement
                            on validation loss to wait for
                            before stopping training.
        """

        self.model = model
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.optimizer = optimizer
        self.criterion = criterion
        self.parameters = parameters
        self.n_epochs = n_epochs
        self.patience = patience

    def _do_train(self,
                  model,
                  loaders,
                  optimizer,
                  criterion,
                  weighted):

        """
        Args:
            model (nn.Module): Model.
            loader (Sampler): Generator of n_train EEG samples for training.
            optimizer (optimizer): Optimizer.
            criterion (Loss): Loss function.
            weigthed (Bool): True if the loader if weighted, False otherwise.
        """

        # Training loop
        model.train()
        device = next(model.parameters()).device
        train_loss = list()
        y_pred_all, y_true_all = list(), list()

        iter_loader = [iter(loader) for loader in loaders]

        # Loop on training samples
        for batch_x, batch_y in iter_loader[0]:
            if len(loaders) > 1:
                batch_x_list = [batch_x]
                batch_y_list = [batch_y]

                for id in len(loaders[1:]):
                    batch_x, batch_y = next(iter_loader[id])
                    batch_x_list.append(batch_x)
                    batch_y_list.append(batch_y)

                batch_x = np.concatenate(batch_x_list, axis=0)
                batch_y = np.concatenate(batch_y_list, axis=0)

            optimizer.zero_grad()

            batch_x = batch_x.to(torch.float).to(device=device)
            batch_y = batch_y.to(torch.float).to(device=device)
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            y_pred_all.append(output.detach().cpu().numpy())
            y_true_all.append(batch_y.detach().cpu().numpy())

        y_pred = np.concatenate(y_pred_all)
        y_true = np.concatenate(y_true_all)

        y_pred_binary = np.zeros(len(y_pred))
        y_pred_binary[np.where(y_pred > 0.5)[0]] = 1

        perf = f1_score(y_true, y_pred_binary)
        return np.mean(train_loss), perf

    def _validate(self,
                  model,
                  loader,
                  criterion):

        # Validation loop
        model.eval()
        device = next(model.parameters()).device

        val_loss = np.zeros(len(loader))
        y_pred_all, y_true_all = list(), list()

        # Loop in validation samples
        with torch.no_grad():
            for idx_batch, (batch_x, batch_y) in enumerate(loader):
                batch_x = batch_x.to(torch.float).to(device=device)
                batch_y = batch_y.to(torch.float).to(device=device)
                output, _ = model.forward(batch_x)

                loss = criterion(output, batch_y)
                val_loss[idx_batch] = loss.item()

                y_pred_all.append(output.cpu().numpy())
                y_true_all.append(batch_y.cpu().numpy())

        y_pred = np.concatenate(y_pred_all)
        y_true = np.concatenate(y_true_all)

        y_pred_binary = np.zeros(len(y_pred))
        y_pred_binary[np.where(y_pred > 0.5)[0]] = 1
        perf = f1_score(y_true, y_pred_binary)
        return np.mean(val_loss), perf

    def train(self):

        """ Training function.

        Returns:
            best_model (nn.Module): The model that led to the best prediction
                                    on the validation
                                    dataset.
        history (list): List of dictionnaries containing training history
                        (loss, accuracy, etc.).
        """

        history = list()
        best_valid_loss = np.inf
        best_model = copy.deepcopy(self.model)
        print(
            "epoch \t train_loss \t valid_loss \t train_perf \t valid_perf"
        )
        print("-" * 80)

        for epoch in range(1, self.n_epochs + 1):

            train_loss, train_perf = self._do_train(
                self.model,
                self.loader_train,
                self.optimizer,
                self.criterion,
            )

            valid_loss, valid_perf = self._validate(self.model,
                                                    self.loader_val,
                                                    self.criterion)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "train_perf": train_perf,
                    "valid_perf": valid_perf,
                }
            )

            print(
                f"{epoch} \t {train_loss:0.4f} \t {valid_loss:0.4f} \t"
                f"{train_perf:0.4f} \t {valid_perf:0.4f}"
            )

            if valid_loss < best_valid_loss:
                print(f"best val loss {best_valid_loss:.4f} "
                      "-> {valid_loss:.4f}")
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(self.model)
                waiting = 0
            else:
                waiting += 1

            # Early stopping
            if self.patience is None:
                best_model = copy.deepcopy(self.model)
            else:
                if waiting >= self.patience:
                    print(f"Stop training at epoch {epoch}")
                    print(f"Best val loss : {best_valid_loss:.4f}")
                    break

        return best_model, history
