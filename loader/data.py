#!/usr/bin/env python

"""
This script is used to recover data and create ground truth.

Usage: type "from data import <class>" to use one of its classes.

Contributors: Ambroise Odonnat and Theo Gnassounou.
"""

import glob
import os
import scipy.io

import numpy as np

from loguru import logger
from os import listdir
from os.path import isfile, join
from utils.utils_ import get_spike_events


class Data:

    def __init__(self, path_root, event_annotation, channel_types,
                 single_channel, sample_frequence):

        """
        Args:
            path_root (str): Path to data.
            event_annotation (str): Event annotation. Example: 'spikeandwave'.
            channel_types (list): Types of selected channels. Example: ['EEG'].
            single_channel (bool): If True, select only the channel on
                                   which event annotation is done.
            sample_frequence (int): Sample frequence of the data.
        """

        self.path_root = path_root
        self.event_annotation = event_annotation
        self.channel_types = channel_types
        self.single_channel = single_channel
        self.sample_frequence = sample_frequence

    def get_trial(self, trial_fname, channel_fname, event_annotation,
                  channel_types, single_channel):

        """ Recover trials of data. Trials with bad channels contain
            the event 'BAD' and are discarded from the experiment.

        Args:
            trial_fname (str): Path to trial file (matlab dictionnary).
            channel_fname (str): Path to channel file (matlab dictionnary).
            event_annotation (str): Event annotation. Example: 'spikeandwave'.
            channel_types (list): Types of selected channels. Example: ['EEG'].
            single_channel (bool): If True, select only the channel on
                                   which event annotation is done.

        Returns:
            data (array): Trial of dimension [n_channels x n_time_points].
            n_spikes (int): Number of spikes in the trial.
            spike_time_points (array): Spike events times.
            times (array): Time points.
            bad_trial (int): If > 0, trial is discarded.
        """

        # Load the trial and corresponding channels
        trial = scipy.io.loadmat(trial_fname)
        channel_mat = scipy.io.loadmat(channel_fname, chars_as_strings=1)

        # Recover epileptic spike events and times
        n_spikes = 0
        spike_time_points = []
        annotated_channel = 0
        bad_trial = 0
        if len(trial['Events']):
            for iEvent in range(len(trial['Events'][0])):
                event = trial['Events'][0][iEvent]
                if event['label'][0] == event_annotation:
                    n_spikes += len(event['times'][0])
                    spike_time_points = event['times'][0]
                    annotated_channel = event['channels'][0][0][0][0][0]
                elif event['label'][0] == 'BAD':
                    bad_trial += 1

        # Select channels
        if single_channel & (annotated_channel != 0):
            past_channel = annotated_channel
            i = 0
            unchanged = (annotated_channel == past_channel)
            inferior = (i < channel_mat['Channel'].shape[1])
            while unchanged & inferior:
                channel_name = channel_mat['Channel'][0, i]['Name'].tolist()[0]
                if channel_name.casefold() == annotated_channel.casefold():
                    annotated_channel = i
                i += 1
        else:
            selected_channels = []
            for i in range(channel_mat['Channel'].shape[1]):
                channel_type = channel_mat['Channel'][0, i]['Type'].tolist()[0]
                if channel_type in channel_types:
                    selected_channels.append(i)
            annotated_channel = np.random.choice(selected_channels)

        # Recover data and time points
        if single_channel:
            data = np.asarray(trial['F'][annotated_channel], dtype='float64')
        else:
            data = np.asarray(trial['F'][selected_channels], dtype='float64')
        times = trial['Time'][0]

        return data, n_spikes, spike_time_points, times, bad_trial

    def get_dataset(self, folder, channel_fname, event_annotation,
                    channel_types, sample_frequence, single_channel):

        """ Get trials with corresponding labels and spike events array
            (1 when a spike occurs and 0 elsewhere).

        Args:
            folder (list): List of paths to trial files (matlab dictionnaries).
            channel_fname (str): Path to channel file (matlab dictionnary).
            event_annotation (str): Event annotation. Example: 'spikeandwave'.
            channel_types (list): Types of selected channels. Example: ['EEG'].
            single_channel (bool): If True, select only the channel on
                                   which event annotation is done.

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
        for trial_fname in folder:
            dataset = self.get_trial(trial_fname, channel_fname,
                                     event_annotation, channel_types,
                                     single_channel)
            data, n_spike, spike_time_points, times, bad_trial = dataset

            # label = 1 if at least one spike occurs, label = 0 otherwise
            n_spike = int((n_spike > 0))

            # Append data and labels if trial not bad
            if bad_trial == 0:
                all_data.append(data)
                all_n_spikes.append(n_spike)

                # Mark time point as 1 if a spike occurs, 0 otherwise.
                spike_events = get_spike_events(spike_time_points, len(times),
                                                sample_frequence)
                all_spike_events.append(spike_events)

        # Stack Dataset along axis 0
        all_data = np.stack(all_data, axis=0)
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

    def get_all_datasets(self, path_root, event_annotation,
                         channel_types, sample_frequence,
                         single_channel):

        """ Recover data and create labels.

        Args:
            path_root (str): Root path to data.
            channel_fname (str): Path to channel file (matlab dictionnary).
            event_annotation (str): Event annotation. Example: 'spikeandwave'.
            channel_types (list): Types of selected channels. Example: ['EEG'].
            single_channel (bool): If True, select only the channel on
                                   which event annotation is done.

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
            if not item.startswith('.'):
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
                    channel_fname = glob.glob(path + "channel_ctf_acc1.mat",
                                              recursive=True)[0]
                    folder.remove(channel_fname)
                    dataset = self.get_dataset(folder, channel_fname,
                                               event_annotation, channel_types,
                                               sample_frequence,
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

        return self.get_all_datasets(self.path_root, self.wanted_event_label,
                                     self.wanted_channel_type,
                                     self.sample_frequence,
                                     self.binary_classification)
