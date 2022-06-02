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

from torch.nn import BCELoss
from torch.optim import Adam

from models.architectures import RNN_self_attention, STT
from models.training import make_model
from loader.dataloader import pad_loader
from loader.data import Data
from utils.evaluation import score

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_parser():

    """ Set parameters for the experiment."""

    parser = argparse.ArgumentParser(
        "Spike detection", description="Spike detection using attention layer"
    )
    parser.add_argument("--method", type=str, default="RNN_self_attention")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--path_data", type=str, default="../IvadomedNifti/")

    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()
path_root = args.path_data
method = args.method
batch_size = args.batch_size
# path_root = args.path_root
# single_channel = args.single_channel
# path_config = args.path_config

# # Recover config dictionnary
# with open(path_config) as f:
#     config = json.loads(f.read())

# # Recover params
# seed = config['seed']
# method = config['method']
# model = method['params']
# batch_size = config['batch_size']
# n_epochs = config['n_epochs']
# patience = config['patience']
# sample_frequence = config['sample_frequence']

if method == 'RNN_self_attention':
    single_channel = True
else:
    single_channel = False

dataset = Data(path_root, 'spikeandwave', single_channel)
all_dataset = dataset.all_datasets()

assert method in ("RNN_self_attention", "transformers_classification",
                  "transformers_detection")

results = []
data = all_dataset[0]
labels = all_dataset[1]
subject_ids = np.asarray(list(data.keys()))
for test_subject_id in subject_ids:

    data_train = []
    labels_train = []
    data_train = []
    train_subject_ids = np.delete(subject_ids,
                                  np.where(subject_ids == test_subject_id))
    val_subject_id = np.random.choice(train_subject_ids)
    train_subject_ids = np.delete(subject_ids,
                                  np.where(subject_ids == val_subject_id))

    print('Validation on: {}, '
          'test on: {}'.format(test_subject_id,
                               val_subject_id))

    # Training dataloader
    train_data = []
    for id in train_subject_ids:
        sessions_trials = data[id]
        for trials in sessions_trials:
            train_data.append(trials)
    train_labels = []
    for id in train_subject_ids:
        sessions_events = labels[id]
        for events in sessions_events:
            train_labels.append(events)

    # Z-score normalization
    target_mean = np.mean([np.mean(data) for data in train_data])
    target_std = np.mean([np.std(data) for data in train_data])
    train_data = [np.expand_dims((data-target_mean) / target_std,
                                 axis=1)
                  for data in train_data]
    loaders_train = pad_loader(train_data,
                               train_labels,
                               batch_size,
                               True,
                               0)

    # Validation dataloader
    val_data = []
    sessions_trials = data[val_subject_id]
    for trials in sessions_trials:
        val_data.append(trials)
    val_labels = []
    sessions_events = labels[val_subject_id]
    for events in sessions_events:
        val_labels.append(events)

    # Z-score normalization
    val_data = [np.expand_dims((data-target_mean) / target_std,
                axis=1)
                for data in val_data]
    loader_val = pad_loader(val_data,
                            val_labels,
                            batch_size,
                            False,
                            0)

    # Test dataloader
    test_data = []
    sessions_trials = data[test_subject_id]
    for trials in sessions_trials:
        test_data.append(trials)
    test_labels = []
    sessions_events = labels[test_subject_id]
    for events in sessions_events:
        test_labels.append(events)

    # Z-score normalization
    test_data = [np.expand_dims((data-target_mean) / target_std,
                                axis=1)
                 for data in test_data]
    loader_test = pad_loader(test_data,
                             test_labels,
                             batch_size,
                             False,
                             0)

    input_size = loaders_train.dataset[0][0].shape[1]

    if method == "RNN_self_attention":
        archi = RNN_self_attention(input_size=1)
    else:
        archi = STT()

    model = model.to(device)

    lr = 1e-3  # Learning rate
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=0)

    criterion = BCELoss()

    model = make_model(archi,
                       loaders_train,
                       loader_val,
                       optimizer,
                       criterion,
                       n_epochs=10,
                       patience=10)

    # Train Model
    best_model, history = model.train()

    if not os.path.exists("../results"):
        os.mkdir("../results")

    if args.save_model:
        model_dir = "../results/{}".format(model)
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        torch.save(
            best_model.state_dict(),
            model_dir / "model_{}_{}".format(method, test_subject_id),
        )

    # Compute test performance and save it
    acc, f1, precision, recall = score(best_model, loader_test)

    results.append(
        {
            "method": method,
            "test_subj_id": test_subject_id,
            "acc": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    )

    results_path = (
        "../results/csv"
    )
    if not os.path.exists(results_path):
        os.mkdir(results_path)

    df_results = pd.DataFrame(results)
    df_results.to_csv(
        os.path.join(results_path,
                     "accuracy_results_spike_detection_method-{}_{}-subjects.csv".format(method, len(subject_ids))))
