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
from utils.model import fukumori2021RNN

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "Spike detection", description="Spike detection using attention layer"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--method", type=str, default="Fukumori")
    parser.add_argument("--batch-size", type=int, default=16)
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

dataset = Data(path_data, False, 'spikeandwave', 'EEG', 100, True)
all_dataset = dataset.all_datasets()

n_epochs = 100
patience = 10

assert method in ("Fukumori")

results = []
data = all_dataset[0]
subject_ids = np.asarray(list(data.keys()))
labels = all_dataset[1]
for test_subject_id in subject_ids:
    
    data_train = []
    labels_train = []
    data_train = []
    train_subject_ids = np.delete(subject_ids, np.where(subject_ids == test_subject_id))
    val_subject_id = np.random.choice(train_subject_ids)
    train_subject_ids = np.delete(subject_ids, np.where(subject_ids == val_subject_id))

    print('Validation on: {}, '
          'test on: {}'.format(test_subject_id,
                               val_subject_id))

    for id in train_subject_ids:
        sessions_trials = data[id]
        data_train.append(np.concatenate(sessions_trials, axis=0))
        sessions_events = labels[id]
        labels_train.append(np.concatenate(sessions_events))
    
    data_train = np.concatenate(data_train, axis=0)     
    labels_train = np.concatenate(labels_train, axis=0)
    data_val = np.concatenate(data[val_subject_id], axis=0)
    labels_val = np.concatenate(labels[val_subject_id], axis=0)
    data_test = np.concatenate(data[test_subject_id], axis=0)
    labels_test = np.concatenate(labels[test_subject_id], axis=0)

    data_train = signal.resample(data_train, 512, axis=1)
    data_train = data_train[:, :, np.newaxis]
    data_test = signal.resample(data_test, 512, axis=1)
    data_test = data_test[:, :, np.newaxis]
    data_val = signal.resample(data_val, 512, axis=1)
    data_val = data_val[:, :, np.newaxis]


    mean = data_train.mean()
    std = data_train.std()
    data_train -= mean
    data_train /= std
    mean = data_val.mean()
    std = data_val.std()
    data_val -= mean
    data_val /= std
    mean = data_test.mean()
    std = data_test.std()
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
        model = fukumori2021RNN(input_size=input_size)

    model = model.to(device)

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
    df_results.to_csv(os.path.join(results_path,"accuracy_results_spike_detection_method-{}_{}-subjects.csv".format(method, len(subject_ids))))
