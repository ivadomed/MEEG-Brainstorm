#!/opt/anaconda3/bin/python

"""
This script is used to get raw and projected data with corresponding labels .

Usage: type "from get_data import <function>" to use one of its functions.

Contributors: Ambroise Odonnat.
"""

import scipy.io
import os

import numpy as np

from common_spatial_pattern import csp


class Data:
    
    def __init__(self, folder, channel_fname, wanted_event_label,\
                    list_channel_type, binary_classification, selected_rows):
        
        """    
        Args:
            folder (list): List of paths to trial files (matlab dictionnaries),
            channel_fname (str): Path to channel file (matlab dictionnary),
            wanted_event_label (str): Annotation of wanted event,
            list_channel_type (list): List of the types of channels we want ,
            binary_classification (bool): Labellize trials in two classes as seizure/no seizure 
                                          instead of taking the number of seizures as label.
        """
            
        self.folder = folder
        self.channel_fname = channel_fname
        self.wanted_event_label = wanted_event_label
        self.list_channel_type = list_channel_type
        self.binary_classification = binary_classification
        self.selected_rows = selected_rows
              

    def get_trial(self, trial_fname, channel_fname, wanted_event_label, wanted_channel_type):

        """
        Get single trial of data.

        Args:
            trial_fname (str): Path to trial file (matlab dictionnary),
            channel_fname (str): Path to channel file (matlab dictionnary),
            wanted_event_label (str): Annotation wanted,
            wanted_channel_type (list): List of types of channels wanted.

        Returns:
            tuple: data (array): Trial of size (n_channel)x(n_sample_points)),
                   label (int): Number of seizures in the trial,
                   spikeTimePoints (array): Spike times,
                   times (array): Time points.
        """

        # Load trial and channels
        trial = scipy.io.loadmat(trial_fname)
        channelMat = scipy.io.loadmat(channel_fname, chars_as_strings=1)

        # Get wanted channels
        wanted_channels = []
        for i in range(channelMat['Channel'].shape[1]):
            channel_type = channelMat['Channel'][0, i]['Type'].tolist()[0]
            if channel_type == wanted_channel_type:
                wanted_channels.append(i)

        # Get signals        
        F = trial['F'][wanted_channels]
        times = trial['Time'][0]

        # Count seizure events and recover spikes' times
        count_seizures = 0
        spikeTimePoints=[]
        event_label = trial['Events']['label'][0][0][0]
        if  event_label == wanted_event_label:
            count_seizures += trial['Events']['times'][0][0].shape[1]
            spikeTimePoints = trial['Events']['times'][0][0][0]


        data, label = F, count_seizures

        return data, label, spikeTimePoints, times


    def get_dataset(self, folder, channel_fname, wanted_event_label, list_channel_type,\
                    binary_classification):

        """
        Compute Common Spatial Pattern matrix W.

        Args:
            folder (list): List of paths to trial files (matlab dictionnaries),
            channel_fname (str): Path to channel file (matlab dictionnary),
            wanted_event_label (str): Annotation of wanted event,
            list_channel_type (list): List of the types of channels we want,
            binary_classification (bool): Labellize trials in two classes as seizure/no seizure,
                                          instead of taking the number of seizures as label.

        Returns:
            tuple: allData (dictionnary): Keys are number of channels and values are the trials stacked on new axis 0,
                   allLabels (dictionnary): Keys are number of channels and values are array of corresponding labels,
                   allSpikeTimePoints (dictionnary): Keys are number of channels and values are array of corresponding spike times,
                   allTimes (dictionnary): Keys are number of channels and values are array of time points,
                   Example: allData[50][0] is a trial of size 50x(n_sample_points),
                            allLabels[50][0] is its corresponding label.

        """

        allData = {}
        allLabels = {}
        allSpikeTimePoints = {}
        allTimes = {}

        # Loop on trials
        for trial_fname in folder:
            for wanted_channel_type in list_channel_type:
                data, label, spikeTimePoints, times = self.get_trial(trial_fname, channel_fname,\
                                                                     wanted_event_label, wanted_channel_type)

                # label = 1 with at least one seizure occurs, label = 0 otherwise
                if binary_classification:
                    label = int((label > 0))

                # Append data and labels for trials with same number of electrode channels n_channel
                n_channel = data.shape[0]
                if not (n_channel in allData):
                    allData[n_channel] = [data]
                    allLabels[n_channel] = [label]
                    allSpikeTimePoints[n_channel] = [spikeTimePoints]
                    allTimes[n_channel] = [times]
                else:
                    allData[n_channel].append(data)
                    allLabels[n_channel].append(label)
                    allSpikeTimePoints[n_channel].append(spikeTimePoints)
                    allTimes[n_channel].append(times)

        # Stack Dataset along axis 0
        for key in allData.keys():
            allData[key] = np.stack(allData[key], axis = 0)
            allLabels[key] = np.asarray(allLabels[key])
            allSpikeTimePoints[key] = np.asarray(allSpikeTimePoints[key], dtype = 'O')
            allTimes[key] = np.asarray(allTimes[key], dtype = 'O')
        return allData, allLabels, allSpikeTimePoints, allTimes
    
    
    def get_projected_data(self, folder, channel_fname, wanted_event_label,\
                           list_channel_type, binary_classification, selected_rows):

        """
        Get dataset and labels after Common Spatial Pattern algorithm.
        All projected trials have same dimension (Nr)x(n_sample_points) with N = n_classes and r = selected_rows.

        Args:
            folder (list): List of paths to trial files (matlab dictionnaries),
            channel_fname (str): Path to channel file (matlab dictionnary),
            wanted_event_label (str): Annotation of wanted event,
            list_channel_type (list): List of the types of channels we want ,
            binary_classification (bool): Labellize trials in two classes as seizure/no seizure 
                                          instead of taking the number of seizures as label.

        Returns:
            tuple: csp_allDatas (array): Trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points),
                   allLabels (array): Corresponding labels,
                   allSpikeTimePoints (array): Corresponding spike times,
                   allTimes (array): Time points.
        """

        # Get dataset, labels, spike time points and time points
        allData, allLabels, allSpikeTimePoints, allTimes = self.get_dataset(folder,channel_fname, wanted_event_label,\
                                                                       list_channel_type, binary_classification)
        csp_allData = {}

        # Loop on the number of electrode channels
        for key in allData.keys():
            assert key in allLabels.keys(), " missing key in dictionnary Labels"

            # Recover data and labels for trials with n_channels = key
            data, labels = allData[key], allLabels[key]

            # Compute corresponding CSP
            W = csp(data, labels, selected_rows)

            # Apply spatial filter W
            csp_allData[key] = np.einsum('cd,ndt -> nct', W, data) 

        # Convert in lists and concatenate
        csp_allData = np.concatenate(list(csp_allData.values()))
        allLabels = np.concatenate(list(allLabels.values()))
        allSpikeTimePoints = np.concatenate(list(allSpikeTimePoints.values()))
        allTimes = np.concatenate(list(allTimes.values()))

        return csp_allData, allLabels, allSpikeTimePoints, allTimes
    
    def csp_data(self):

        """
        Get dataset and labels after Common Spatial Pattern algorithm.
        All projected trials have same dimension (Nr)x(n_sample_points) with N = n_classes and r = selected_rows.
        Returns:
            tuple: csp_allDatas (array): Trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points),
                   allLabels (array): Corresponding labels,
                   allSpikeTimePoints (array): Corresponding spike times,
                   allTimes (array): Time points.
        """
        
        return self.get_projected_data(self.folder, self.channel_fname, self.wanted_event_label,\
                                       self.list_channel_type, self.binary_classification, self.selected_rows)


