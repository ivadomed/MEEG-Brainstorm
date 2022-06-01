#!/usr/bin/env python

"""
This script is used to recover data and create ground truth.

Usage: type "from data import <class>" to use one of its classes.

Contributors: Ambroise Odonnat and Theo Gnassounou.

This script is used to recover data and create ground truth.
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

from utils.utils_ import get_spike_events


def get_spike_events(spike_time_points,
                     n_time_points):

    """
    Compute array of dimension [n_time_points] with 1
    when a spike occurs and 0 otherwise.

    Args:
        spike_time_points (array): Contains time points
                                   when a spike occurs if any.
        n_time_points (int): Number of time points.
        freq (int): Sample frequence of the EEG/MEG signals.

    Returns:
        spike_events (array): Array of dimension [n_time_points] containing 1
                              when a spike occurs and 0 otherwise.
    """

    spike_events = np.zeros(n_time_points)
    for time in spike_time_points:
        spike_events[time] = 1

    return spike_events.astype(int)


class Data:

    def __init__(self,
                 path_root,
                 wanted_event_label,
                 single_channel):

        """
        Args:
            path_root (str): Path to subjects data.
            label_position (bool): If True, event labels are in first position.
                          Else, they are in last position.
            wanted_event_label (str): Annotation of wanted event.
                                      Example: 'saw_EST' -> peaks of spikes.
            wanted_channel_type (list): List of the types of channels wanted.
                                        Example: ['EEG'].
            sample_frequence (int): Sample frequence of the data.
            binary_classification (bool): If True, we label trials with no
                                          seizure/seizure as 0/1.
        """

        self.path_root = path_root
        self.wanted_event_label = wanted_event_label
        self.single_channel = single_channel

    def get_trial(self,
                  raw,
                  events,
                  wanted_event_label,
                  wanted_channels,
                  single_channel):

        """ Recover as numpy array a trial with corresponding number of spikes,
            spike events times and time points. Trials with bad channels
            contain the event 'BAD' and must be discarded from the experiment.
        Args:
            trial_fname (str): Path to trial file (matlab dictionnary).
            channel_fname (str): Path to channel file (matlab dictionnary).
            wanted_event_label (str): Annotation of wanted event.
                                      Example: 'saw_EST' -> peaks of spikes.
            wanted_channel_type (list): List of the types of channels wanted.
                                        Example: ['EEG'].
        Returns:
            data (array): Trial of dimension [n_channels x n_time_points].
            label (int): Number of seizures in the trial.
            spikeTimePoints (array): Spike events times.
            times (array): Time points.
            bad_trial (int): If nonzero, trial is further discarded.
        """

        # Load the trial and corresponding channels

        events_name = events[1].keys()
        # Count seizure events and recover spikes events times
        count_spikes = 0
        spike_time_points = []
        bad_trial = 0

        if wanted_event_label in events_name:
            events_points = np.where(events[0][:, 2]
                                     == events[1][wanted_event_label])
            count_spikes += len(events_points)
            spike_time_points = events[0][events_points, 0][0]

        if 'BAD' in events_name:
            bad_trial = 1

        # Recover data and time points
        if single_channel:
            data = raw[:][0][wanted_channels, :-49]
        else:
            data = raw[:][0][:, :-49]

        times = raw.times[:-49]
        sfreq = raw.info['sfreq']
        return data, count_spikes, spike_time_points, times, bad_trial, sfreq

    def get_dataset(self,
                    folder,
                    wanted_event_label,
                    single_channel):

        """ Get trials with corresponding labels and spike events array
            (1 when a spike occurs and 0 elsewhere).
        Args:
            folder (list): List of paths to trial files (matlab dictionnaries).
            channel_fname (str): Path to channel file (matlab dictionnary).
            wanted_event_label (str): Annotation of wanted event.
                                      Example: 'saw_EST' -> peaks of spikes.
            wanted_channel_type (list): List of the types of channels we want.
                                        Example: ['EEG'].
            sample_frequence (int): Sample frequence of the data.
            binary_classification (bool): If True, we label trials with no
                                          seizure/seizure as 0/1.
        Returns:
            all_data (array): Trials of dimension
                              [n_trials x n_channels x n_time_points].
            all_labels (array): Labels of dimension [n_trials].
            all_spike_events (array): Spike events of dimension
                                      [n_trials x n_time_points].
        """

        all_data = []
        all_n_spikes = []
        all_spike_events = []

        # Loop on trials
        wanted_channels = []

        if single_channel:

            for trial_fname in folder:
                try:
                    raw = mne.io.read_raw_edf(trial_fname, preload=False,
                                              stim_channel=None, verbose=False)
                    events = mne.events_from_annotations(raw, verbose=False)
                    ch_names = raw.info.ch_names
                    for event in events[1].keys():
                        len_string_event = len(wanted_event_label)
                        match = (event[-len_string_event:]
                                 == wanted_event_label)
                        possible = (len(event) > len_string_event)
                        if match & possible:
                            ID = 'EEG ' + event[:2].upper()
                            wanted_channels.append(np.where(np.array(ch_names)
                                                            == ID)[0][0])
                except ValueError:
                    continue

            wanted_channels = np.unique(wanted_channels)
            if len(wanted_channels) == 0:
                wanted_channels = [np.random.randint(0, len(ch_names))]

        for trial_fname in folder:
            try:

                raw = mne.io.read_raw_edf(trial_fname, preload=False,
                                          stim_channel=None, verbose=False)
                events = mne.events_from_annotations(raw, verbose=False)
                dataset = self.get_trial(raw,
                                         events,
                                         wanted_event_label,
                                         wanted_channels,
                                         single_channel)

                data, n_spike, spike_time_points, times, bad_trial, sfreq = dataset

                # Apply binary classification
                # label = 1 if at least one spike occurs, label = 0 otherwise

                n_spike = int((n_spike > 0))

                # Append data and labels from each good trial
                if bad_trial == 0:
                    all_data.append(data)
                    all_n_spikes.append(n_spike)

                    # Get vector with 1 when a spike occurs and 0 elsewhere
                    N = len(times)
                    spike_events = get_spike_events(spike_time_points, N)
                    all_spike_events.append(spike_events)

            except ValueError:
                continue

        # Stack Dataset along axis 0
        all_data = np.stack(all_data, axis=0)

        if single_channel:
            ntrials, nchan, ntime = all_data.shape
            # Each channels become a trials
            all_data = all_data.reshape(ntrials*nchan, ntime)
            all_data = scipy.signal.resample(all_data, 512, axis=1)

        all_n_spikes = np.asarray(all_n_spikes)
        all_spike_events = np.asarray(all_spike_events, dtype='int64')

        """ Label creation: n_classes different number of spikes.
            Order them by increasing order in an array of dimension
            [n_classes]: class i has label i.
            Example: trials have only 1 or 3 spikes in the dataset,
                     labels will be 0 and 1 respectively.
        """

        unique_n_spike = np.unique(all_n_spikes)
        all_labels = np.asarray([np.where(unique_n_spike == s)[0][0]
                                 for s in all_n_spikes])
        logger.info("Label creation: number of spikes {} mapped on "
                    "labels {}".format(np.unique(all_n_spikes),
                                       np.unique(all_labels)))

        return all_data, all_labels, all_spike_events

    def get_all_datasets(self,
                         path_root,
                         wanted_event_label,
                         single_channel):

        """ Recover data and create labels.
        Args:
            path_root (str): Root path to data.
            wanted_event_label (str): Annotation of wanted event.
                                      Example: 'saw_EST' -> peaks of spikes.
            wanted_channel_type (list): List of the types of channels we want.
                                        Example: ['EEG'].
            sample_frequence (int): Sample frequence of the data.
            binary_classification (bool): If True, we label trials with no
                                          seizure/seizure as 0/1.
        Returns:
            all_data (dict): Keys -> subjects; values -> trials of dimension
                             [n_trials x n_channels x n_time_points].
            all_labels (dict):  Keys -> subjects; values -> labels
                                of dimension [n_trials].
            all_spike_events (dict):  Keys -> subjects; values -> spike events
                                      of dimension [n_trials x n_time_points].
        """

        all_data = {}
        all_labels = {}
        all_spike_events = {}
        for item in os.listdir(path_root):

            if not isfile(path_root + item):
                logger.info("Recover data for {}".format(item))
                subject_data, subject_labels, subject_spike_events = [], [], []
                subject_path = path_root+item+'/'
                sessions = [f.path for f in os.scandir(subject_path)
                            if f.is_dir()]

                # Recover trials, labels and spike events

                for i in range(len(sessions)):
                    path = sessions[i] + '/'
                    folder = [path + f for f in listdir(path)
                              if isfile(join(path, f))]
                    dataset = self.get_dataset(folder,
                                               wanted_event_label,
                                               single_channel)
                    data, labels, spike_events = dataset
                    subject_data.append(data)
                    subject_labels.append(labels)
                    subject_spike_events.append(spike_events)

                # Recover trials for each subject
                all_data[item] = subject_data
                all_labels[item] = subject_labels
                all_spike_events[item] = subject_spike_events

        return all_data, all_labels, all_spike_events

    def all_datasets(self):

        """ Recover data and create labels."""

        return self.get_all_datasets(self.path_root,
                                     self.wanted_event_label,
                                     self.single_channel)
