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

from utils.utils_ import get_spike_events, get_spike_windows


class Data:

    def __init__(self,
                 path_root,
                 wanted_event_label,
                 n_windows,
                 single_channel):

        """
        Args:
            path_root (str): Path to data.
            wanted_event_label (str): Annotation of wanted event.
                                      Default: 'spikeandwave'.
            n_windows (int): Number of time windows.
            single_channel (bool): If True, only select the channels with
                                   the event annotation one.
        """

        self.path_root = path_root
        self.wanted_event_label = wanted_event_label
        self.n_windows = n_windows
        self.single_channel = single_channel

    def get_trial(self,
                  raw_trial,
                  events,
                  wanted_event_label,
                  annotated_channels,
                  single_channel):

        """ Recover trials in .edf format.

        Args:
            raw_trial (.edf file): Trial of EEG data in .edf format.
            events (dict): Annotated events on the trial.
            wanted_event_label (str): Annotation of wanted event.
                                      Default: 'spikeandwave'.
            annotated_channels (list): List of channels with spike annotation
                                       to select if single_channel is True.
            single_channel (bool): If True, only select the channels with
                                   the event annotation one.
        Returns:
            data (array): Trial of dimension [n_channels x n_time_points].
            cout_spikes (int): Number of seizures in the trial.
            spike_time_points (array): Spike events times.
            times (array): Time points.
            bad_trial (int): If nonzero, trial is further discarded.
            sfreq (int): Sample frequence of the trial.
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
            data = raw_trial[:][0][annotated_channels, :-49]
        else:
            data = raw_trial[:][0][:, :-49]

        times = raw_trial.times[:-49]
        sfreq = raw_trial.info['sfreq']

        return data, count_spikes, spike_time_points, times, bad_trial, sfreq

    def get_dataset(self,
                    folder,
                    wanted_event_label,
                    n_windows,
                    single_channel):

        """ Recover trials in .edf format as BIDS dataset.
            Binary labels: 0 if no spike in the trial, 1 otherwise.
            Spike events times: Array of size [n_time_points] with 1
                                when spike occurs, 0 otherwise.
        Args:
            folder (list): Folder with a subject data.
            wanted_event_label (str): Annotation of wanted event.
                                      Default: 'spikeandwave'.
            n_windows (int): Number of time windows.
            single_channel (bool): If True, only select the channels with
                                   the event annotation one.
        Returns:
            all_data (array): Trials of dimension
                              [n_trials x n_channels x n_time_points].
            all_labels (array): Labels of dimension [n_trials].
            all_spike_events (array): Spike events times of dimension
                                      [n_trials x n_time_points].
            sfreq (int): Sample frequence of the trial.
        """

        all_data = []
        all_n_spikes = []
        all_spike_events = []

        # Loop on trials
        annotated_channels = []

        if single_channel:
            for trial_fname in folder:
                try:
                    raw_trial = mne.io.read_raw_edf(trial_fname, preload=False,
                                                    stim_channel=None,
                                                    verbose=False)
                    events = mne.events_from_annotations(raw_trial,
                                                         verbose=False)
                    ch_names = raw_trial.info.ch_names
                    for event in events[1].keys():
                        len_string_event = len(wanted_event_label)
                        match = (event[-len_string_event:]
                                 == wanted_event_label)
                        possible = (len(event) > len_string_event)
                        if match & possible:
                            ID = 'EEG ' + event[:2].upper()
                            position_channels = np.where(
                                                np.array(ch_names) == ID)[0]
                            if len(position_channels) != 0:
                                annotated_channels.append(position_channels[0])

                except ValueError:
                    continue

            annotated_channels = np.unique(annotated_channels)
            if len(annotated_channels) == 0:
                annotated_channels = [np.random.randint(0, len(ch_names))]

        for trial_fname in folder:
            try:
                raw_trial = mne.io.read_raw_edf(trial_fname, preload=False,
                                                stim_channel=None,
                                                verbose=False)
                events = mne.events_from_annotations(raw_trial, verbose=False)
                dataset = self.get_trial(raw_trial,
                                         events,
                                         wanted_event_label,
                                         annotated_channels,
                                         single_channel)

                data, n_spike, spike_time_points, times, bad, sfreq = dataset

                # Apply binary classification
                # label = 1 if at least one spike occurs, label = 0 otherwise
                n_spike = int((n_spike > 0))

                # Append data and labels from each good trial
                if bad == 0:
                    all_data.append(data)
                    all_n_spikes.append(n_spike)

                    # Get vector with 1 when a spike occurs and 0 elsewhere
                    N = len(times)
                    spike_events = get_spike_events(spike_time_points,
                                                    N, sfreq)
                    spike_windows = get_spike_windows(spike_events, n_windows)
                    all_spike_events.append(spike_windows)

            except ValueError:
                continue

        # Stack Dataset along axis 0
        all_data = np.stack(all_data, axis=0)

        if single_channel:
            ntrials, nchan, ntime = all_data.shape

            # Each channels become a trials
            all_data = all_data.reshape(ntrials*nchan, ntime)
            all_data = scipy.signal.resample(all_data, 512, axis=1)
            all_n_spikes = all_n_spikes*nchan

        all_n_spikes = np.asarray(all_n_spikes)
        all_spike_events = np.array(all_spike_events, dtype='int64')

        logger.info("Label creation: number of spikes {} mapped on "
                    "labels {}".format(np.unique(all_n_spikes),
                                       np.unique(all_n_spikes)))

        return all_data, all_n_spikes, all_spike_events, sfreq

    def get_all_datasets(self,
                         path_root,
                         wanted_event_label,
                         n_windows,
                         single_channel):

        """ Recover data and create labels for all subjects.
        Args:
            path_root (str): Path to data.
            wanted_event_label (str): Annotation of wanted event.
                                      Default: 'spikeandwave'.
            n_windows (int): Number of time windows.
            single_channel (bool): If True, only select the channels with
                                   the event annotation one.
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
                                               n_windows,
                                               single_channel)
                    data, labels, spike_events, sfreq = dataset
                    subject_data.append(data)
                    subject_labels.append(labels)
                    subject_spike_events.append(spike_events)

                # Recover trials for each subject
                all_data[item] = subject_data
                all_labels[item] = subject_labels
                all_spike_events[item] = subject_spike_events

        return all_data, all_labels, all_spike_events, sfreq

    def all_datasets(self):

        """ Recover data and create labels."""

        return self.get_all_datasets(self.path_root,
                                     self.wanted_event_label,
                                     self.n_windows,
                                     self.single_channel)
