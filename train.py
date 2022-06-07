#!/usr/bin/env python

"""
This script is used to train models.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import argparse
import json
import os
from platform import architecture
import numpy as np
import pandas as pd
import torch

from loguru import logger
from torch.nn import BCELoss
from torch.optim import Adam

from models.architectures import RNN_self_attention, STT
from models.training import make_model
from loader.dataloader import Loader
from loader.data import Data
from utils.cost_sensitive_loss import get_criterion
from utils.utils_ import define_device, reset_weights


def get_parser():

    """ Set parameters for the experiment."""

    parser = argparse.ArgumentParser(
        "Spike detection", description="Spike detection using attention layer"
    )
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--method", type=str, default="RNN_self_attention")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_windows", type=int, default=1)
    parser.add_argument("--path_root", type=str, default="../BIDSdataset/")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--average", type=str, default="weighted")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cost_sensitive", action="store_true")
    parser.add_argument("--lambd", type=float, default=0.0)

    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()
path_root = args.path_root
n_windows = args.n_windows
n_epochs = args.n_epochs
save = args.save
method = args.method
batch_size = args.batch_size
num_workers = args.num_workers
balanced = args.balanced
cost_sensitive = args.cost_sensitive
lambd = args.lambd
average = args.average

gpu_id = 0
weight_decay = 0
lr = 1e-3  # Learning rate
patience = 10

available, device = define_device(gpu_id)

# Define loss
criterion = BCELoss().to(device)
train_criterion = get_criterion(criterion,
                                cost_sensitive,
                                lambd)

if method == 'RNN_self_attention':
    single_channel = True
else:
    single_channel = False

dataset = Data(path_root, 'spikeandwave', single_channel)
all_dataset = dataset.all_datasets()

assert method in ("RNN_self_attention", "transformer_classification",
                  "transformer_detection")
logger.info(f"Method used: {method}")

# Recover results
results = []

data, labels, spikes, sfreq = dataset.all_datasets()
subject_ids = np.asarray(list(data.keys()))

# TODO z-score with only trin data

# Training dataloader
train_labels = []
train_data = []
train_spikes = []
for id in subject_ids:
    train_data.append(data[id])
    train_labels.append(labels[id])
    train_spikes.append(spikes[id])

# Z-score normalization
target_mean = np.mean([np.mean([np.mean(data) for data in data_id])
                        for data_id in train_data])
target_std = np.mean([np.mean([np.std(data) for data in data_id])
                        for data_id in train_data])
train_data = [[np.expand_dims((data-target_mean) / target_std, axis=1)
                for data in data_id] for data_id in train_data]

for seed in range(5):
    # Dataloader
    if method == "transformer_detection":

        # Labels are the spike events times
        train_loader = Loader(train_data,
                              train_spikes,
                              balanced=balanced,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              split_dataset=True)
    else:

        # Label is 1 with a spike occurs in the trial, 0 otherwise
        train_loader = Loader(train_data,
                              train_labels,
                              balanced=balanced,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              split_dataset=True)
    train_dataloader, val_dataloader, test_dataloader = train_loader.load()

    # Define architecture
    if method == "RNN_self_attention":
        architecture = RNN_self_attention()
    elif method == "transformer_classification":
        architecture = STT(n_windows=n_windows)
    elif method == "transformer_detection":
        detection = True
        architecture = STT(n_windows=n_windows, detection=detection)
    architecture.apply(reset_weights)

    # Define optimizer
    optimizer = Adam(architecture.parameters(), lr=lr,
                        weight_decay=weight_decay)

    # Define training pipeline
    architecture = architecture.to(device)

    model = make_model(architecture,
                       train_dataloader,
                       val_dataloader,
                       test_dataloader,
                       optimizer,
                       train_criterion,
                       criterion,
                       n_epochs=n_epochs,
                       patience=patience,
                       average=average)

    # Train Model
    history = model.train()

    if not os.path.exists("../results"):
        os.mkdir("../results")

    # Compute test performance and save it
    acc, f1, precision, recall, f1_macro, precision_macro, recall_macro = model.score()

    results.append(
        {
            "method": method,
            "balance": balanced,
            "fold": seed,
            "acc": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1_macro,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
        }
    )

    if save:

        # Save results file as csv
        if not os.path.exists("../results"):
            os.mkdir("../results")

        results_path = (
            "../results/csv"
        )
        if not os.path.exists(results_path):
            os.mkdir(results_path)

        df_results = pd.DataFrame(results)
        df_results.to_csv(
            os.path.join(results_path,
                            "accuracy_results_spike_detection_method-{}"
                            "_balance-{}_{}"
                            "-subjects.csv".format(method, balanced,
                                                len(subject_ids))))
