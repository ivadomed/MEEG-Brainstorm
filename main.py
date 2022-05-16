import pickle
import argparse
import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from sklearn.metrics import balanced_accuracy_score

from utils.model import fukumori2021RNN
from utils.training import train
from utils.evaluation import score
from utils.data import Data
from config import RESULTS_PATH, MODEL_PATH, DATA_PATH

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "Spike detection", description="Domain Adaptation on sleep EEG."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-subjects-s", type=int, default=30)
    parser.add_argument("--n-subjects-t", type=int, default=30)
    parser.add_argument("--method", type=str, default="Fukumori")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--parameters", type=float, nargs="+", default=[1e-2, 1e-1, 1e-3, 1]
    )

    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()  # you can modify this namespace for quick iterations

# load data filtered
expe_seed = args.seed
method = args.method
parameters = args.parameters
batch_size = args.batch_size
sfreq = 100  # Sampling frequency

dataset = Data(path_root, 'spiekandwave', 'eeg', True)
all_dataset = dataset.all_datasets()

n_epochs = 100
patience = 10

assert method in ("Fukumori")

num_workers = 0  # Number of processes to use for the data loading process; 0 is the main Python process

results = []
results_per_class = []

loaders_train, loader_valid, loader_test = 


if method == "Fukumori":
    model = fukumori2021RNN()

lr = 1e-3  # Learning rate
optimizer = Adam(params=model.paremeters, lr=lr)

criterion = CrossEntropyLoss()

# Train Model
best_model, history = train(
    model,
    method,
    loaders_train,
    loader_valid,
    optimizer,
    criterion,
    parameters,
    n_epochs,
    patience,
)

if args.save_model:
    model_dir = MODEL_PATH / f"{method}"
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
    RESULTS_PATH
    / "csv"
    / f"accuracy_results_spike_detection_method-{method}_seed-{expe_seed}.csv"
)
df_results = pd.DataFrame(results)
df_results.to_csv(results_path)
