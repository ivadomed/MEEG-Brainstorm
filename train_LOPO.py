#!/usr/bin/env python

"""
This script is used to train models.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import argparse
import os

import numpy as np
import pandas as pd

from loguru import logger
from torch import nn
from torch.optim import Adam

from models.architectures import *
from models.training import make_model
from loader.dataloader import Loader
from loader.data import Data
from utils.cost_sensitive_loss import get_criterion
from utils.learning_rate_warmup import NoamOpt
from utils.utils_ import define_device, get_pos_weight, reset_weights
from utils.select_subject import select_subject
from augmentation import AffineScaling, FrequencyShift, ChannelsShuffle


def get_parser():

    """ Set parameters for the experiment."""

    parser = argparse.ArgumentParser(
        "Spike detection", description="Spike detection"
    )
    parser.add_argument("--path_root",
                        type=str,
                        default="../BIDSdataset/Epilepsy_dataset/")
    parser.add_argument("--method", type=str, default="RNN_self_attention")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_subjects", type=int, default=5)
    parser.add_argument("--selected_subjects", type=str, nargs="+", default=[])
    parser.add_argument("--weight_loss", action="store_true")
    parser.add_argument("--cost_sensitive", action="store_true")
    parser.add_argument("--lambd", type=float, default=1e-3)
    parser.add_argument("--len_trials", type=float, default=2)
    parser.add_argument("--transform", action="store_true")
    parser.add_argument("--patience", type=int, default=10)

    return parser


# Recover paths and method
parser = get_parser()
args = parser.parse_args()
path_root = args.path_root
method = args.method
save = args.save
warmup = args.warmup
batch_size = args.batch_size
num_workers = args.num_workers
n_epochs = args.n_epochs
n_subjects = args.n_subjects
selected_subjects = args.selected_subjects
weight_loss = args.weight_loss
cost_sensitive = args.cost_sensitive
lambd = args.lambd
len_trials = args.len_trials
transform = args.transform
patience = args.patience

# Recover params
lr = 1e-3  # Learning rate
weight_decay = 0
gpu_id = 0

# Define device
available, device = define_device(gpu_id)

# Define loss
criterion = nn.BCEWithLogitsLoss().to(device)

# Recover results
results = []
mean_acc, mean_f1, mean_precision, mean_recall = 0, 0, 0, 0
steps = 0

# Recover dataset
assert method in ("EEGNet", "EEGNet_1D", "GTN",
                  "RNN_self_attention", "STT")
logger.info(f"Method used: {method}")
if method in ["EEGNet_1D", "RNN_self_attention"]:
    single_channel = True
else:
    single_channel = False

path_subject_info = (
    "../results/info_subject_{}".format(len_trials)
)
if selected_subjects == []:
    selected_subjects = select_subject(n_subjects,
                                       path_subject_info,
                                       path_root,
                                       len_trials)

dataset = Data(path_root,
               'spikeandwave',
               selected_subjects,
               len_trials=len_trials)

data, labels, annotated_channels = dataset.all_datasets()
subject_ids = np.asarray(list(data.keys()))

# Define transform for data augmentation
if transform:

    affine_scaling = AffineScaling(
        probability=.5,  # defines the probability of modifying the input
        a_min=0.7,
        a_max=1.3,
    )

    frequency_shift = FrequencyShift(
        probability=.5,  # defines the probability of modifying the input
        sfreq=128,
        max_delta_freq=.2
    )

    channels_shuffle = ChannelsShuffle(
        probability=.5,  # defines the probability of modifying the input
        p_shuffle=0.2
    )

    transforms = [affine_scaling, frequency_shift, channels_shuffle]
else:
    transforms = None

# Apply Leave-One-Patient-Out strategy

""" Each subject is chosen once as test set while the model is trained
    and validate on the remaining ones.
"""
for gen_seed in range(5):
    np.random.seed(gen_seed)
    seed_list = [np.random.randint(0, 100)
                 for _ in range(len(selected_subjects))]
    for i, test_subject_id in enumerate(subject_ids):
        seed = seed_list[i]
        # Labels are the spike events times
        loader = Loader(data,
                        labels,
                        annotated_channels,
                        single_channel=single_channel,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        transforms=transforms,
                        subject_LOPO=test_subject_id,
                        seed=seed,
                        )

        train_loader, val_loader, test_loader, train_labels = loader.load()

        # Define architecture
        if method == "EEGNet":
            n_time_points = len(data[subject_ids[0]][0][0][0])
            architecture = EEGNet()
            warmup = False
        if method == "EEGNet_1D":
            architecture = EEGNet_1D()
            warmup = False
        elif method == "GTN":
            n_time_points = len(data[subject_ids[0]][0][0][0])
            architecture = GTN(n_time_points=n_time_points)
        elif method == "RNN_self_attention":
            n_time_points = len(data[subject_ids[0]][0][0][0])
            architecture = RNN_self_attention(n_time_points=n_time_points)
        elif method == "STT":
            n_time_points = len(data[subject_ids[0]][0][0][0])
            architecture = STT(n_time_points=n_time_points)
        architecture.apply(reset_weights)

        # Define training loss
        if weight_loss:
            pos_weight = get_pos_weight(train_labels).to(device)
            train_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            train_criterion = train_criterion.to(device)
        else:
            train_criterion = criterion
        train_criterion = get_criterion(train_criterion,
                                        cost_sensitive,
                                        lambd)
        # Define optimizer
        optimizer = Adam(architecture.parameters(), lr=lr,
                         weight_decay=weight_decay)
        warm_optimizer = NoamOpt(optimizer)

        # Define training pipeline
        architecture = architecture.to(device)
        model = make_model(architecture,
                           train_loader,
                           val_loader,
                           test_loader,
                           optimizer,
                           warmup,
                           warm_optimizer,
                           train_criterion,
                           criterion,
                           single_channel=single_channel,
                           n_epochs=n_epochs,
                           patience=patience)

        # Train Model
        history = model.train()

        # Compute test performance and save it
        acc, f1, precision, recall = model.score()
        results.append(
            {
                "method": method,
                "warmup": warmup,
                "weight_loss": weight_loss,
                "cost_sensitive": cost_sensitive,
                "len_trials": len_trials,
                "test_subject_id": test_subject_id,
                "transform": transform,
                "fold": seed,
                "acc": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
            }
        )
        mean_acc += acc
        mean_f1 += f1
        mean_precision += precision
        mean_recall += recall
        steps += 1

        if save:

            # Save results file as csv
            if not os.path.exists("../results"):
                os.mkdir("../results")

            results_path = (
                "../results/csv_LOPO"
            )
            if not os.path.exists(results_path):
                os.mkdir(results_path)

            results_path = os.path.join(results_path,
                                        "results_LOPO_spike_detection_{}"
                                        "-subjects.csv".format(
                                                        len(subject_ids))
                                        )
            df_results = pd.DataFrame(results)
            results = []

            if os.path.exists(results_path):
                df_old_results = pd.read_csv(results_path)
                df_results = pd.concat([df_old_results, df_results])
            df_results.to_csv(results_path, index=False)

    print("Mean accuracy \t Mean F1-score \t Mean precision \t Mean recall")
    print("-" * 80)
    print(
        f"{mean_acc/steps:0.4f} \t {mean_f1/steps:0.4f} \t"
        f"{mean_precision/steps:0.4f} \t {mean_recall/steps:0.4f}\n"
    )
