#!/opt/anaconda3/bin/python

"""
This script is used to visualize EEG and MEG signals. 
Final visualization is done through the MNE framework `<https://mne.tools/stable/index.html>`
Usage: type "from data_visualization import <function>" to use one of its functions.

Contributors: Ambroise Odonnat.
"""

import mne
import random
import scipy.io

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def mat_to_df(trial_fname, channel_fname, channel_type, wanted_event_label, output = False):
    
    """
    Convert EEG/MEG file from .mat to pandas dataframe.
    
    Args:
        trial_fname (str): Path to trial file (matlab dictionnary),
        channel_fname (str): Path to channel file (matlab dictionnary),
        channel_type (list): List of types of channels wanted,
        output (bool): If output, show pd dataframe head.

    Returns:
            df (pd dataframe): EEG/MEG trial.
    """
    
    # Load mat-file
    trial = scipy.io.loadmat(trial_fname)    
    channelMat = scipy.io.loadmat(channel_fname, chars_as_strings=1)
    data = trial['F']          
    channels = channelMat['Channel']  

    channels_list = []
    for channel_array in channels['Name'][0]:
        channels_list.append(channel_array[0])

    # Recover spikes' times
    spikeTimePoints=[]
    event_label = trial['Events']['label'][0][0][0]
    if  event_label == wanted_event_label:
        spikeTimePoints = trial['Events']['times'][0][0][0]
        
    # Recover channels of channel_type
    wanted_channels = []
    for i in range(len(channels_list)):
        if channels['Type'][0][i] in channel_type:
            wanted_channels.append(i)
    
    # Recover corresponding signals and channels' names
    data = data[wanted_channels]
    channels_list = np.array(channels_list)[wanted_channels]
    df = pd.DataFrame(data,
                    index=channels_list)

    df = df.T

    # Remove columns that do not change value
    df = df.loc[:, (df != df.iloc[0]).any()]

    if output:
        display(df.head())

    return df, spikeTimePoints



def mne_object(data, freq, channel_type):

    """
    Create an mne array of data .
    
    Args:
        data (pd dataframe): EEG/MEG trial,
        freq (int): Sample rate of the data,
        channel_type (list): List of types of channels wanted.
    
    Returns:
            raw (mne array): Data with metadata info about channels.
    """
        
    if channel_type == ['EEG']:
        ch_types = ['eeg']*data.shape[-1]
    elif channel_type == ['MEG','MEG REF']:
        ch_types = ['ref_meg']*data.shape[-1]
    info = mne.create_info(ch_names = list(data.columns), 
                         sfreq = freq, 
                         ch_types = ch_types)

    # Transpose the data
    data_T = data.transpose()

    # Create raw mne object
    raw = mne.io.RawArray(data_T, info)

    return raw



def plot_signals_from_df(trial_df, spikeTimePoints, channel_type, wanted_event_label,\
                         freq, n_channel, scaling, highpass, lowpass):

    """
    Plot EEG/MEG trial.
    
    Args:
        trial_df (pd dataframe): EEG/MEG tria,
        freq (int): Sample rate of the data,
        scaling (float): zoom out of the plot,
        highpass (float): frequence for highpass filter,
        lowpass (float): frequence for lowpass filter.

    """
    
    raw = mne_object(trial_df,freq, channel_type)
    
    # Mark spike times
    if spikeTimePoints.size :
        N = spikeTimePoints.shape[0]
        new_annotations = mne.Annotations(spikeTimePoints,\
                                          [1e-15 for i in range(N)],\
                                          [wanted_event_label for i in range(N)])
        raw.set_annotations(new_annotations)
        
    plot_kwargs = {
    'scalings': dict(eeg=scaling),   # zooms the plot out
    'highpass': highpass,              # filters out low frequencies
    'lowpass': lowpass,                # filters out high frequencies
    'show_scrollbars': False,
    'show': True}
    raw.plot(**plot_kwargs, n_channels = n_channel) 

    
    
def plot_signals_from_file(trial_fname, channel_fname, channel_type,\
                           wanted_event_label, freq, n_channel, scaling, highpass, lowpass):

    """
    Plot EEG/MEG trial.
    
    Args:
        trial_fname (str): Path to trial file (matlab dictionnary),
        channel_fname (str): Path to channel file (matlab dictionnary),
        channel_type (list): List of types of channels wanted,
        freq (int): Sample rate of the data,
        scaling (float): zoom out of the plot,
        highpass (float): frequence for highpass filter,
        lowpass (float): frequence for lowpass filter.

    """
    

    m, spikeTimePoints = mat_to_df(trial_fname, channel_fname, channel_type, wanted_event_label, False)
    raw = mne_object(m,freq, channel_type)
    
    # Mark spike times
    if spikeTimePoints.size :
        N = spikeTimePoints.shape[0]
        new_annotations = mne.Annotations(spikeTimePoints,\
                                          [1e-15 for i in range(N)],\
                                          [wanted_event_label for i in range(N)])
        raw.set_annotations(new_annotations)
    
    # Plot EEG/MEG trial
    plot_kwargs = {
    'scalings': dict(eeg=scaling),   # zooms the plot out
    'highpass': highpass,              # filters out low frequencies
    'lowpass': lowpass,                # filters out high frequencies
    'show_scrollbars': False,
    'show': True}
    raw.plot(**plot_kwargs, n_channels = n_channel)

    

def csp_visualization(allData, csp_allData, allLabels, allSpikeTimePoints, allTimes,\
                         label_1, label_2, width, height, seed = 0, show_spikes = False):
    
    """
    Plots of signals of all channels before and after Common Spatial Pattern projection.
    
    Args:
        allData (array): Raw trials (n_trials)x(n_channels)x(n_sample_points),
        csp_allDatas (array): Trials after CSP algorithm (n_trials)x(Nr)x(n_sample_points),
        allLabels (array): Corresponding labels,
        allSpikeTimePoints (array): Corresponding spike times,
        allTimes (array): Time points.
    
    """
 
    # First set of signals
    random.seed(seed)
    index_1 = random.choice(np.where(allLabels[50] == label_1)[0]) 
    X_EEG1, X_MEG1 =  allData[50][index_1], allData[274][index_1] 
    X_cspEEG1, X_cspMEG1 = csp_allData[index_1], csp_allData[655 + index_1]
    spikeTimePoints1 = allSpikeTimePoints[50][index_1]
    times = allTimes[50][index_1]


    # Second set of signals
    index_2 = random.choice(np.where(allLabels[50] == label_2)[0]) 
    X_EEG2, X_MEG2 =  allData[50][index_2], allData[274][index_2] 
    X_cspEEG2, X_cspMEG2 = csp_allData[index_2], csp_allData[655 + index_2]
    spikeTimePoints2 = allSpikeTimePoints[50][index_2]


    data = [X_EEG1, X_cspEEG1, X_EEG2, X_cspEEG2, X_MEG1, X_cspMEG1, X_MEG2, X_cspMEG2]

    # Plot signals
    titles = ['EEG','csp_EEG','EEG','csp_EEG','MEG','csp_MEG','MEG','csp_MEG']
    axes=[]
    rows, cols = 4,2 # As we have 8 signals to plot
    fig=plt.figure(1, figsize=(width,height))
    for a in range(rows*cols):
        axes.append(fig.add_subplot(rows, cols, a+1))

        for x in data[a]:
            plt.plot(times,x)
            
        if a in [0,1,4,5]: 
            if label_1 != 0:
                subplot_title=(""+titles[a]+'_seizure_'+str(index_1))
                axes[-1].set_title(subplot_title)
                if show_spikes:
                    for spike in spikeTimePoints1:
                        plt.axvline(x=spike, color = 'black')
            else:
                subplot_title=(""+titles[a]+'_'+str(index_1))
                axes[-1].set_title(subplot_title)
        else:
            if label_2 != 0:          
                subplot_title=(""+titles[a]+'_seizure_'+str(index_2))
                axes[-1].set_title(subplot_title)
                if show_spikes:
                    for spike in spikeTimePoints2:
                        plt.axvline(x=spike, color = 'black')
            else:
                subplot_title=(""+titles[a]+'_'+str(index_2))
                axes[-1].set_title(subplot_title)

    fig.tight_layout()    
    plt.show()
    print('index trial 1: %s '%index_1, 'index trial 2: %s '%index_2,'\n',\
          'EEG shape ', 'before CSP : ',X_EEG1.shape,' after CSP: ', X_cspEEG1.shape,'\n',\
         'MEG shape ', 'before CSP : ',X_MEG1.shape,' after CSP: ', X_cspMEG1.shape)
    