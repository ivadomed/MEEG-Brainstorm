#!/usr/bin/env python

"""
This script is used to train and evaluate models.

Usage: type "from training import <class>" to use one of its classes.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import copy
import torch

import numpy as np

from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, accuracy_score

from utils.utils_ import get_next_batch


class make_model():

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 test_loader,
                 optimizer,
                 criterion,
                 n_epochs,
                 patience=None):

        """
        Args:
            model (nn.Module): Model.
            train_loader (Dataloader): Loader of EEG samples for training.
            val_loader (Dataloader): Loader of EEG samples for validation.
            test_loader (Dataloader): Loader of EEG samples for test.
            optimizer (optimizer): Optimizer.
            criterion (Loss): Loss function.
            n_epochs (int): Maximum number of epochs to run.
            patience (int): Indicates how many epochs without improvement
                            on validation loss to wait for
                            before stopping training.
        """

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.n_epochs = n_epochs
        self.patience = patience

    def _do_train(self,
                  model,
                  loaders,
                  optimizer,
                  criterion):

        """
        Train model.
        Args:
            model (nn.Module): Model.
            loaders (Sampler): Generator of n_train EEG samples for training.
            optimizer (optimizer): Optimizer.
            criterion (Loss): Loss function.

        Returns:
            train_loss (float): Mean loss on the loaders.
            perf (float): Mean F1-score on the loaders.
        """

        # Training loop
        model.train()
        device = next(model.parameters()).device
        train_loss = list()
        all_preds, all_labels = list(), list()

        iter_loader = [iter(loader) for loader in loaders]

        # Loop on training samples
        for batch_x, batch_y in iter_loader[0]:
            if len(loaders) > 1:
                batch_x_list = [batch_x]
                batch_y_list = [batch_y]
                for id in range(len(loaders[1:])):
                    batch_x, batch_y = get_next_batch(id, iter_loader, loaders)
                    batch_x_list.append(batch_x)
                    batch_y_list.append(batch_y)
                batch_x = torch.cat(batch_x_list, dim=0)
                batch_y = torch.cat(batch_y_list, dim=0)
            batch_x = batch_x.to(torch.float).to(device=device)
            batch_y = batch_y.to(torch.float).to(device=device)

            # Optimizer
            optimizer.zero_grad()

            # Forward
            output, _ = model(batch_x)
            loss = criterion(output, batch_y)

            # Backward
            loss.backward()
            optimizer.step()

            # Recover loss and prediction
            train_loss.append(loss.item())
            all_preds.append(output.detach().cpu().numpy())
            all_labels.append(batch_y.detach().cpu().numpy())

        # Recover binary prediction
        y_pred = np.concatenate(all_preds)
        y_pred_binary = 1 * (y_pred > 0.5)
        y_true = np.concatenate(all_labels)

        # Recover mean loss and F1-score
        train_loss = np.mean(train_loss)
        perf = f1_score(y_true, y_pred_binary, average='weighted',
                        zero_division=0)

        return train_loss, perf

    def _validate(self,
                  model,
                  loader,
                  criterion):

        """
        Evaluate model on validation set.
        Args:
            model (nn.Module): Model.
            loader (Sampler): Generator of n_val EEG samples for validation.
            criterion (Loss): Loss function.

        Returns:
            val_loss (float): Mean loss on the loader.
            perf (float): Mean F1-score on the loader.
        """

        # Validation loop
        model.eval()
        device = next(model.parameters()).device

        val_loss = np.zeros(len(loader[0]))
        all_preds, all_labels = list(), list()

        # Loop in validation samples
        with torch.no_grad():
            for idx_batch, (batch_x, batch_y) in enumerate(loader[0]):
                batch_x = batch_x.to(torch.float).to(device=device)
                batch_y = batch_y.to(torch.float).to(device=device)

                # Forward
                output, _ = model.forward(batch_x)

                # Recover loss and prediction
                loss = criterion(output, batch_y)
                val_loss[idx_batch] = loss.item()
                all_preds.append(output.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())

        # Recover binary prediction
        y_pred = np.concatenate(all_preds)
        y_pred_binary = 1 * (y_pred > 0.5)
        y_true = np.concatenate(all_labels)

        # Recover mean loss and F1-score
        val_loss = np.mean(val_loss)
        perf = f1_score(y_true, y_pred_binary, average='weighted',
                        zero_division=0)

        return val_loss, perf

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
        best_val_loss = np.inf
        self.best_model = copy.deepcopy(self.model)
        print(
            "epoch \t train_loss \t val_loss \t train_perf \t val_perf"
        )
        print("-" * 80)

        for epoch in range(1, self.n_epochs + 1):

            train_loss, train_perf = self._do_train(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
            )

            val_loss, val_perf = self._validate(self.model,
                                                self.val_loader,
                                                self.criterion)

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_perf": train_perf,
                    "valid_perf": val_perf,
                }
            )

            print(
                f"{epoch} \t {train_loss:0.4f} \t {val_loss:0.4f} \t"
                f"{train_perf:0.4f} \t {val_perf:0.4f}"
            )

            if val_loss < best_val_loss:
                print(f"best val loss {best_val_loss:.4f} "
                      f"-> {val_loss:.4f}")
                best_val_loss = val_loss
                self.best_model = copy.deepcopy(self.model)
                waiting = 0
            else:
                waiting += 1

            # Early stopping
            if self.patience is None:
                self.best_model = copy.deepcopy(self.model)
            else:
                if waiting >= self.patience:
                    print(f"Stop training at epoch {epoch}")
                    print(f"Best val loss : {best_val_loss:.4f}")
                    break

        return history

    def score(self):

        # Compute performance on test set
        self.best_model.eval()
        device = next(self.best_model.parameters()).device

        all_preds, all_labels = list(), list()
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader[0]:
                batch_x = batch_x.to(torch.float).to(device=device)
                batch_y = batch_y.to(torch.float).to(device=device)

                # Forward
                output, _ = self.best_model.forward(batch_x)

                # Recover prediction
                all_preds.append(output.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())

        # Recover binary prediction
        y_pred = np.concatenate(all_preds)
        y_pred_binary = 1 * (y_pred > 0.5)
        y_true = np.concatenate(all_labels)

        # Recover performances
        acc = accuracy_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary, average='weighted',
                      zero_division=0)
        precision = precision_score(y_true, y_pred_binary,
                                    average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred_binary, average='weighted',
                              zero_division=0)
        print(f"Accuracy on test {acc} "
              f"F1-score on test {f1} "
              f"Precision on test {precision} "
              f"Recall on test {recall} ")
        
        return acc, f1, precision, recall
