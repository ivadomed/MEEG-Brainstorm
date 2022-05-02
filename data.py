#!/usr/bin/env python

"""
This script is used to split dataset between training, validation and testing
and project data with CSP matrix computed on training data.

Usage: type "from data import <class>" to use one of its classes.

Contributors: Ambroise Odonnat.
"""


import os
import scipy.io

import numpy as np

from loguru import logger
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split

from common_spatial_pattern import common_spatial_pattern
from utils import get_spike_events


class Data:
    
    def __init__(self, path_root, wanted_event_label, wanted_channel_type, binary_classification):
        
        """    
        Args:
            path_root (str): Path to subjects data.
            wanted_event_label (str): Annotation of wanted event. 
                                      Example: 'saw_EST' corresponds to peaks of spikes.
            wanted_channel_type (list): List of the types of channels wanted. Example: ['EEG'].
            binary_classification (bool): If True, we label trials with no seizure/seizure as 0/1. 
            selected_rows (int): Number of first and last selected rows in each sub-spatial
                                 filter to create global spatial filter.
        """
            
        self.path_root = path_root
        self.wanted_event_label = wanted_event_label
        self.wanted_channel_type = wanted_channel_type
        self.binary_classification = binary_classification

        
    def get_trial(self, trial_fname, channel_fname, wanted_event_label, wanted_channel_type):

        """
        Recover as numpy array a trial with corresponding number of spikes, spike events times and time points.

        Args:
            trial_fname (str): Path to trial file (matlab dictionnary).
            channel_fname (str): Path to channel file (matlab dictionnary).
            wanted_event_label (str): Annotation of wanted event. 
                                      Example: 'saw_EST' corresponds to peaks of spikes.
            wanted_channel_type (list): List of the types of channels wanted. Example: ['EEG'].

        Returns:
            data (array): Trial of dimension [n_channels x n_time_points].
            label (int): Number of seizures in the trial.
            spikeTimePoints (array): Spike events times.
            times (array): Time points.
        """

        # Load the trial and corresponding channels
        trial = scipy.io.loadmat(trial_fname)
        channel_mat = scipy.io.loadmat(channel_fname, chars_as_strings=1)

        # Select the wanted type of channels
        wanted_channels = []
        for i in range(channel_mat['Channel'].shape[1]):
            channel_type = channel_mat['Channel'][0, i]['Type'].tolist()[0]
            if channel_type in wanted_channel_type:
                wanted_channels.append(i)

        # Recover data and time points        
        F = trial['F'][wanted_channels]
        times = trial['Time'][0]

        # Count seizure events and recover spikes events times
        count_spikes = 0
        spike_time_points=[]
        event_label = trial['Events']['label'][0][0][0]
        if  event_label == wanted_event_label:
            count_spikes += trial['Events']['times'][0][0].shape[1]
            spike_time_points = np.round(trial['Events']['times'][0][0][0],2) # In seconds

        data, n_spike = np.asarray(F, dtype='float64'), count_spikes

        return data, n_spike, spike_time_points, times

    
    def get_dataset(self, folder, channel_fname, wanted_event_label, wanted_channel_type, binary_classification):

        """
        Get trial with corresponding labels and spike events array (1 when a spike occurs and 0 elsewhere).

        Args:
            folder (list): List of paths to trial files (matlab dictionnaries).
            channel_fname (str): Path to channel file (matlab dictionnary).
            wanted_event_label (str): Annotation of wanted event.
                                      Example: 'saw_EST' corresponds to peaks of spikes.
            wanted_channel_type (list): List of the types of channels we want. Example: ['EEG'].
            binary_classification (bool): If True, we label trials with no seizure/seizure as 0/1. 

        Returns:
            all_data (array): Trials of dimension [n_trials x n_channels x n_time_points].
            all_labels (array): Corresponding labels of dimension [n_trials].
            all_spike_events (array): Corresponding spike events of dimension [n_trials x n_time_points].
                                      Example: all_data[0] is a trial of dimension [n_channels x n_time_points].
                                               all_labels[0] is its corresponding label (int value).
                                               all_spike_events[0] is an array of dimension [n_time_points] with 1 when a spike occurs.
        """

        all_data = []
        all_n_spikes = []
        all_spike_events = []
        all_times = []

        # Loop on trials
        for trial_fname in folder:
            data, n_spike, spike_time_points, times = self.get_trial(trial_fname, channel_fname, 
                                                                     wanted_event_label, wanted_channel_type)
            
            # Apply binary classification: label = 1 if at least one spike occurs, label = 0 otherwise
            if binary_classification:
                n_spike = int((n_spike > 0))

            # Append data and labels from each trial 
            all_data.append(data)
            all_n_spikes.append(n_spike)
            
            # Get vector with 1 when a spike occurs and 0 elsewhere
            N = len(times)
            freq = 100 # In our dataset, the sample frequence is 100 Hz.
            spike_events = get_spike_events(spike_time_points, N, freq)
            all_spike_events.append(spike_events)

        # Stack Dataset along axis 0
        all_data = np.stack(all_data, axis = 0)
        all_n_spikes = np.asarray(all_n_spikes)
        all_spike_events = np.asarray(all_spike_events, dtype='int64')
        
        """
        Label creation
        We have n_classes different number of spikes in the dataset.
        Number of spikes ordered by increasing order in an array of dimension [n_classes]. Class i has label i.
        Example: trials have only 1 or 3 spikes in the dataset, labels will be 0 and 1 respectively.
        """
        
        unique_n_spike = np.unique(all_n_spikes)
        all_labels = np.asarray([np.where(unique_n_spike == s)[0][0] for s in all_n_spikes])
        
        # Insure that the labels correspond to the number of spike events if multi-classification
        if binary_classification:
            logger.info("Label creation: No Spike / Spikes mapped on "
                        "labels {}".format(np.unique(all_labels)))  
        else:
            logger.info("Label creation: number of spikes {} mapped on "
                        "labels {}".format(np.unique(all_n_spikes), np.unique(all_labels)))
             
        return all_data, all_labels, all_spike_events
    
    
    def get_all_datasets(self, path_root, wanted_event_label, wanted_channel_type, binary_classification):
        
        all_data = {}
        all_labels = {}
        all_spike_events = {}
        
        for item in os.listdir(path_root):
            if not item.startswith('.'):
                subject_path = path_root+item+'/'
                for subject_item in os.listdir(subject_path):
                    path = subject_path+subject_item
                    if os.path.isfile(path):
                        channel_fname = path
                    if os.path.isdir(path):
                        path += '/'
                        folder = [path+f for f in listdir(path) if isfile(join(path, f))]
                        
                # Get dataset, labels, spike time points and time points
                logger.info("Recover data for {}".format(item)) 
                data, labels, spike_events = self.get_dataset(folder,channel_fname, wanted_event_label, 
                                                                          wanted_channel_type, binary_classification)

                all_data[item] = data
                all_labels[item] = labels
                all_spike_events[item] = spike_events
                
        return all_data, all_labels, all_spike_events
    
    
    def all_datasets(self):
        
        return self.get_all_datasets(self.path_root, self.wanted_event_label,self.wanted_channel_type, self.binary_classification)
    
    
    def get_csp_datasets(self, folder, channel_fname, wanted_event_label, wanted_channel_type,
                         binary_classification, selected_rows, train_size, test_size, shuffle,
                         random_state):

        """
        Compute CSP projection on training data, split dataset into training, validation, test sets and apply CSP projection.
        Each projected trial has dimension [(2Nr) x n_time_points] with N = n_classes and r = selected_rows.
        If it is a binary classification, each trial has dimension [(2r) x n_time_points].

        Args:
            folder (list): List of paths to trial files (matlab dictionnaries).
            channel_fname (str): Path to channel file (matlab dictionnary).
            wanted_event_label (str): Annotation of wanted event.
                                      Example: 'saw_EST' corresponds to peaks of spikes.
            wanted_channel_type (list): List of the types of channels we want. Example: ['EEG'].
            binary_classification (bool): If True, we label trials with no seizure/seizure as 0/1. 
            selected_rows (int): Number of first and last rows selected on each sub-filter.

        Returns:
            n_classes (int): Number of classes in the dataset.
            train_csp_data (array): Contains training trials after CSP algorithm of dimension [n_trials x (2Nr) x n_time_points].
                                    If it is a binary classification, array of dimension [n_trials x (2r) x n_time_points].
            train_labels (array): Corresponding labels.
            train_spike_events (array): Corresponding spike events.
            val_csp_data (array): Contains validation trials after CSP algorithm of dimension [n_trials x (2Nr) x n_time_points].
                                  If it is a binary classification, array of dimension [n_trials x (2r) x n_time_points].
            val_labels (array): Corresponding labels.
            val_spike_events (array): Corresponding spike events.
            test_csp_data (array): Contains testing trials after CSP algorithm of dimension [n_trials x (2Nr) x n_time_points].
                                   If it is a binary classification, array of dimension [n_trials x (2r) x n_time_points].
            test_labels (array): Corresponding labels.
            test_spike_events (array): Corresponding spike events.
        """

        # Get dataset, labels, spike time points and time points
        all_data, all_labels, all_spike_events = self.get_dataset(folder,channel_fname, wanted_event_label, 
                                                                  wanted_channel_type, binary_classification)
        
        # Recover number of classes in the entire dataset
        n_classes = len(np.unique(all_labels))
            
        # Split between training and test sets
        split = train_test_split(all_data, all_labels, all_spike_events, train_size=(1-test_size),
                                 shuffle=shuffle, random_state=random_state)
        all_data, test_data, all_labels, test_labels, all_spike_events, test_spike_events = split
        
        # Split between training and validation sets
        split = train_test_split(all_data, all_labels, all_spike_events, train_size=train_size/(1-test_size),
                                 shuffle=shuffle, random_state=random_state)
        train_data, val_data, train_labels, val_labels, train_spike_events, val_spike_events = split


        # Compute CSP projection on the training set
        self.CSP_projection = common_spatial_pattern(train_data, train_labels, selected_rows)

        # Apply spatial filter CSp_projection on training, validation and test data
        train_csp_data = np.einsum('cd,ndt -> nct', self.CSP_projection, train_data)     
        val_csp_data = np.einsum('cd,ndt -> nct', self.CSP_projection, val_data) 
        test_csp_data = np.einsum('cd,ndt -> nct', self.CSP_projection, test_data) 
        
        # Z-score standardization 
        self.target_mean = np.mean(train_csp_data)
        self.target_std = np.std(train_csp_data)
        train_csp_data = (train_csp_data-self.target_mean) / self.target_std
        val_csp_data = (val_csp_data-self.target_mean) / self.target_std
        test_csp_data = (test_csp_data-self.target_mean) / self.target_std
        
        return (n_classes, train_csp_data, train_labels, train_spike_events,
                val_csp_data, val_labels, val_spike_events,
                test_csp_data, test_labels, test_spike_events)
    
    
    def csp_datasets(self):

        """
        Compute CSP projection on training data, split dataset into training, validation, test sets and apply CSP projection.
        Each projected trial has dimension [(2Nr) x n_time_points] with N = n_classes and r = selected_rows.
        If it is a binary classification, each trial has dimension [(2r) x n_time_points].
        """
        
        return self.get_csp_datasets(self.folder, self.channel_fname, self.wanted_event_label,self.wanted_channel_type,
                                     self.binary_classification, self.selected_rows, self.train_size, self.test_size,
                                     self.shuffle, self.random_state)
    
    

        


