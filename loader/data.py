#!/usr/bin/env python

"""
This script is used to recover EEG data in BIDS dataset with .edf format.
This format is only possible for EEG.

Usage: type "from data import <class>" to use one of its classes.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import os
import scipy.io
import mne
import scipy

import numpy as np

from loguru import logger
from os import listdir
from os.path import isfile, join
from scipy import signal

from utils.utils_ import get_spike_events, get_spike_windows


class Dataset():
    """Returns samples from an mne.io.Raw object along with a target.
    Dataset which serves samples from an mne.io.Raw object along with a target.
    The target is unique for the dataset, and is obtained through the
    `description` attribute.
    Parameters
    ----------
    data : Continuous data.
    labels : labels.
    transform : callable | None
        On-the-fly transform applied to the example before it is returned.
    """
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        if self.transform is not None:
            X = self.transform(X)
        return X, y

    def __len__(self):
        return len(self.x)

class Data:

    def __init__(self,
                 path_root,
                 wanted_event_label,
                 selected_subjects,
                 len_trials=2,
                 sfreq=128,):

        """
        Args:
            path_root (str): Path to data.
            wanted_event_label (str): Annotation of wanted event.
                                      Default: 'spikeandwave'.
            len_trials (float): len of the trials in seconds.
            sfreq (int): Sample frequence of the trial wanted.
            n_windows (int): Number of time windows.
        """

        self.path_root = path_root
        self.wanted_event_label = wanted_event_label
        self.selected_subjects = selected_subjects
        self.len_trials = len_trials
        self.sfreq = sfreq

    def get_trials(self,
                   raw_trial,
                   events,
                   wanted_event_label,
                   len_trials,
                   sfreq):

        """ Recover trials in .edf format.

        Args:
            raw_trial (.edf file): Trial of EEG data in .edf format.
            events (dict): Annotated events on the trial.
            wanted_event_label (str): Annotation of wanted event.
                                      Default: 'spikeandwave'.
            len_trials (float): len of the trials in seconds.
            sfreq (int): Sample frequence of the trial wanted.


        Returns:
            data (array): Trial of dimension
                          [n_trials x n_channels x n_time_points].
            count_spikes (array): Number of spikes in the trials.
            count_bad (array): Number of BAD in the trials.
        """

        # Recover data
        ch_names = raw_trial.info.ch_names

        annotated_channels = []
        # Get the channels where the spikeandwave is annotated
        for event in events[1].keys():
            len_string_event = len(wanted_event_label)
            if event[-(len_string_event+1):] == '_' + wanted_event_label:
                ID = 'EEG '
                i = 0
                while event[i] != '_':
                    ID += event[i]
                    i += 1
                ID = ID.upper()
                position_channels = np.where(
                                    np.array(ch_names) == ID)[0]
                if len(position_channels) != 0:
                    annotated_channels.append(position_channels[0])

        annotated_channels = np.unique(annotated_channels)
        # Get the data

        data = raw_trial[:][0]

        # split the data in trials
        # TODO deal with overlap and clean this script

        len_data = data.shape[1]
        sfreq_ini = raw_trial.info['sfreq']
        n_trials = int(len_data/sfreq_ini / len_trials)
        data = data[:, :int(n_trials*len_trials*sfreq_ini)]
        len_data = data.shape[1]
        times = np.split(np.linspace(0, len_data-1, len_data), n_trials)
        data = np.split(data, n_trials, axis=1)
        data = np.array(data)
        count_spikes = np.zeros(n_trials)
        count_bad = np.zeros(n_trials)
        for event in events[0]:
            try:
                if events[1][wanted_event_label] == event[2]:
                    j = 0
                    while event[0] >= times[j][-1]:
                        j += 1
                    count_spikes[j] += 1
            except KeyError:
                continue
            except IndexError:
                continue

            try:
                if events[1]['BAD'] == event[2]:
                    j = 0
                    while event[0] >= times[j][-1]:
                        j += 1
                    count_bad[j] += 1

            except KeyError:
                continue

        # Resample the data to the frequence wanted
        len_data = data.shape[2]
        data = scipy.signal.resample(data,
                                     int(len_data/sfreq_ini*sfreq),
                                     axis=2)

        return data, count_spikes, count_bad, annotated_channels

    def get_dataset(self,
                    run_fname,
                    wanted_event_label,
                    len_trials,
                    sfreq,):

        """ Recover trials in .edf format as BIDS dataset.
            Binary labels: 0 if no spike in the trial, 1 otherwise.
            Spike events times: Array of size [n_time_points] with 1
                                when spike occurs, 0 otherwise.
        Args:
            folder (list): Folder with a subject data.
            wanted_event_label (str): Annotation of wanted event.
                                      Default: 'spikeandwave'.
            len_trials (float): len of the trials in seconds.
            sfreq (int): Sample frequence of the trial wanted.
            n_windows (int): Number of time windows.

        Returns:
            data (array): Trials of dimension
                              [n_trials x n_channels x n_time_points].
            events (array): Labels of dimension [n_trials].
        """

        raw_run = mne.io.read_raw_edf(run_fname, preload=False,
                                      stim_channel=None,
                                      verbose=False)
        events = mne.events_from_annotations(raw_run, verbose=False)

        dataset = self.get_trials(raw_run,
                                  events,
                                  wanted_event_label,
                                  len_trials,
                                  sfreq)
        data, count_spikes, count_bad, annotated_channels = dataset

        # Apply binary classificatin
        labels = 1*(count_spikes > 0)

        # Remove BAD trials
        good_trials = np.where(count_bad == 0)[0]
        data = data[good_trials]
        labels = labels[good_trials]

        logger.info("Number of spikes {}".format(np.sum(labels)))

        return data, labels, annotated_channels

    def get_all_datasets(self,
                         path_root,
                         wanted_event_label,
                         selected_subjects,
                         len_trials,
                         sfreq,
                         n_windows=1):

        """ Recover data and create labels for all subjects.
        Args:
            path_root (str): Path to data.
            wanted_event_label (str): Annotation of wanted event.
                                      Default: 'spikeandwave'.
            len_trials (float): len of the trials in seconds.
            sfreq (int): Sample frequence of the trial wanted.
            n_windows (int): Number of time windows.

        Returns:
            all_data (dict): Keys -> subjects; values -> list of trials of
                             dimension [n_trials x n_channels x n_time_points].
            all_labels (dict):  Keys -> subjects; values -> list of labels
                                of dimension [n_trials].
            all_spike_events (dict):  Keys -> subjects; values -> list of
                                      spike events of dimension
                                      [n_trials x n_time_points].
            sfreq (int): Sample frequence of the trial.
        """

        all_data = {}
        all_labels = {}
        all_annotated_channels = {}

        if selected_subjects == []:
            selected_subjects = os.listdir(path_root)

        for item in selected_subjects:

            if not isfile(path_root + item):
                logger.info("Recover data for {}".format(item))
                subject_data = []
                subject_labels = []
                subject_annotated_channels = []
                subject_path = path_root+item+'/eeg/'

                run_fnames = [subject_path + f for f in listdir(subject_path)
                              if isfile(join(subject_path, f))]
                for run_fname in run_fnames:

                    dataset = self.get_dataset(run_fname,
                                               wanted_event_label,
                                               len_trials,
                                               sfreq,)
                    data, labels, annotated_channels = dataset
                    subject_data.append(data)
                    subject_labels.append(labels)
                    subject_annotated_channels.append(annotated_channels)

                # Recover trials for each subject
                all_data[item] = subject_data
                all_labels[item] = subject_labels
                all_annotated_channels[item] = subject_annotated_channels

        return all_data, all_labels, all_annotated_channels

    def all_datasets(self):

        """ Recover data and create labels."""

        return self.get_all_datasets(self.path_root,
                                     self.wanted_event_label,
                                     self.selected_subjects,
                                     self.len_trials,
                                     self.sfreq,)
