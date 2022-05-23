import pickle
import argparse
import os
import numpy as np
import pandas as pd
import torch
import scipy

from scipy import signal

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, BCELoss
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from utils.training import train
from utils.evaluation import score
from utils.data_fukumori import Data, SpikeDetectionDataset

import utils.model as model
import utils.training as training

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "Spike detection", description="Domain Adaptation on sleep EEG."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--method", type=str, default="Fukumori")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--path_data", type=str, default="../IvadomedNifti/")

    parser.add_argument(
        "--parameters", type=float, nargs="+", default=[1e-2, 1e-1, 1e-3, 1]
    )

    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()
# load data filtered
expe_seed = args.seed
method = args.method
parameters = [0]
batch_size = args.batch_size
path_data = args.path_data
sfreq = 100  # Sampling frequency

dataset = Data(path_data, 'spikeandwave', 'EEG', True)
all_dataset = dataset.all_datasets()

n_epochs = 1
patience = 10

assert method in ("Fukumori")

results = []

data = all_dataset[0]['Neuropoly_MEEG_database']
data_resample = signal.resample(data, 512, axis=1)
data_resample = data_resample[:, :, np.newaxis]

labels = all_dataset[1]['Neuropoly_MEEG_database']


data_train, data_test, labels_train, labels_test = train_test_split(
                                                                data_resample,
                                                                labels)
data_train, data_val, labels_train, labels_val = train_test_split(data_train,
                                                                  labels_train)

mean = data_train.mean()
std = data_train.std()
data_train -= mean
data_train /= std
data_val -= mean
data_val /= std
data_test -= mean
data_test /= std

dataset_train = SpikeDetectionDataset(data_train, labels_train)
dataset_val = SpikeDetectionDataset(data_val, labels_val)
dataset_test = SpikeDetectionDataset(data_test, labels_test)

loader_train = DataLoader(dataset_train, batch_size=batch_size)
loader_val = DataLoader(dataset_val, batch_size=batch_size)
loader_test = DataLoader(dataset_test, batch_size=batch_size)

input_size = loader_train.dataset[0][0].shape[1]

if method == "Fukumori":
    model = model.fukumori2021RNN(input_size=input_size)


lr = 1e-3  # Learning rate
optimizer = Adam(model.parameters(), lr=lr, weight_decay=0)

criterion = BCELoss()


# Train Model
best_model, history = train(
    model,
    method,
    loader_train,
    loader_val,
    optimizer,
    criterion,
    parameters,
    n_epochs,
    patience,
)

if args.save_model:
    model_dir = 'results' / f"{method}"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    torch.save(
        best_model.state_dict(),
        model_dir / "model_{}".format(method),
    )

# Compute test performance and save it
acc, bal, coh, _, _, _ = score(best_model, loader_test)

results.append(
    {
        "method": method,
        "acc": acc,
        "bal": bal,
        "coh": coh,
        "expe_seed": expe_seed,
    }
)

results_path = (
    'results'
    / "csv"
    / f"accuracy_results_spike_detection_method-{method}_seed-{expe_seed}.csv"
)
df_results = pd.DataFrame(results)
df_results.to_csv(results_path)
