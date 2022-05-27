import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import torch


def score(model, loader):
    # Compute    performance

    model.eval()
    device = next(model.parameters()).device

    y_pred_all, y_true_all = list(), list()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(torch.float).to(device=device)
            batch_y = batch_y.to(torch.float).to(device=device)
            output, _ = model.forward(batch_x)
            y_pred_all.append(output.cpu().numpy())
            y_true_all.append(batch_y.cpu().numpy())

    y_pred = np.concatenate(y_pred_all)
    y_true = np.concatenate(y_true_all)

    y_pred_binary = np.zeros(len(y_pred))
    y_pred_binary[np.where(y_pred > 0.5)[0]] = 1

    acc = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary, average='weighted')
    precision = precision_score(y_true, y_pred_binary, average='weighted')
    recall = recall_score(y_true, y_pred_binary, average='weighted')

    return acc, f1, precision, recall
