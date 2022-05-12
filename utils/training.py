import copy

import numpy as np

from sklearn.metrics import f1_score, accuracy_score

import torch
from torch.nn import CrossEntropyLoss


def _do_train(
    model,
    loader,
    optimizer,
    criterion,
):
    # training loop
    model.train()

    device = next(model.parameters()).device

    train_loss = list()

    len_dataloader = len(loader)
    y_pred_all, y_true_all = list(), list()

    for batch_x, batch_y in loader:

        optimizer.zero_grad()

        batch_x = batch_x.to(torch.float).to(device=device)
        batch_y = batch_y.to(device=device)
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


def _validate(model, loader, criterion):
    # validation loop
    model.eval()
    device = next(model.parameters()).device

    val_loss = np.zeros(len(loader))
    y_pred_all, y_true_all = list(), list()
    with torch.no_grad():
        for idx_batch, (batch_x, batch_y) in enumerate(loader):
            batch_x = batch_x.to(torch.float).to(device=device)
            batch_y = batch_y.to(device=device)
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


def train(
    model,
    method,
    loader,
    loader_valid,
    optimizer,
    criterion,
    parameters,
    n_epochs,
    patience=None,
):
    """Training function.
    Parameters
    ----------
    model : instance of nn.Module
        The model.
    loader : instance of Sampler
        The generator of EEG samples the model has to train on.
        It contains n_train samples
    loader_valid : instance of Sampler
        The generator of EEG samples the model has to validate on.
        It contains n_val samples. The validation samples are used to
        monitor the training process and to perform early stopping
    optimizer : instance of optimizer
        The optimizer to use for training.
    parameters : Parameters of the model.
    n_epochs : int
        The maximum of epochs to run.
    patience : int
        The patience parameter, i.e. how long to wait for the
        validation error to go down.
    metric : None | callable
        Metric to use to evaluate performance on the training and
        validation sets. Defaults to balanced accuracy.
    Returns
    -------
    best_model : instance of nn.Module
        The model that led to the best prediction on the validation
        dataset.
    history : list of dicts
        Training history (loss, accuracy, etc.)
    """
    history = list()
    best_valid_loss = np.inf
    best_model = copy.deepcopy(model)
    print(
        "epoch \t train_loss \t valid_loss \t train_perf \t valid_perf"
    )
    print("-" * 80)

    for epoch in range(1, n_epochs + 1):

        train_loss, train_perf = _do_train(
            model,
            loader,
            optimizer,
            criterion,
        )

        valid_loss, valid_perf = _validate(model, loader_valid, criterion)

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
            print(f"best val loss {best_valid_loss:.4f} -> {valid_loss:.4f}")
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(model)
            waiting = 0
        else:
            waiting += 1

        # model early stopping
        if patience is None:
            best_model = copy.deepcopy(model)
        else:
            if waiting >= patience:
                print(f"Stop training at epoch {epoch}")
                print(f"Best val loss : {best_valid_loss:.4f}")
                break

    return best_model, history
