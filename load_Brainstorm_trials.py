import scipy.io

trial_fname = '/home/nas/Consulting/brainstorm_db/Epilepsy/data/sub-pt0090/sub-pt0090_ses-20210722_task-rest_run-03_meg/data_spaced_out_2_trial837_notch_band_resample_time.mat'
channel_fname = '/home/nas/Consulting/brainstorm_db/Epilepsy/data/sub-pt0090/sub-pt0090_ses-20210722_task-rest_run-03_meg/channel_ctf_acc1.mat'

trial = scipy.io.loadmat(trial_fname)
channelMat = scipy.io.loadmat(channel_fname, chars_as_strings=1)

# Get signals
F = trial['F']  # mChannels x nSamples
time = trial['Time']  # nTimeSamples

# Get channels information
channelNames = []
channelPositions = []
channelTypes = []
for i in range(394):
    channelNames.append(channelMat['Channel'][0, i]['Name'].tolist()[0])  # This holds the signal names
    channelPositions.append(channelMat['Channel'][0, i]['Loc'])  # This holds the signal coordinates. Focus only on the EEG signals that have 3 coordinates
    channelTypes.append(channelMat['Channel'][0, i]['Type'].tolist()[0])  # This holds the information regarding the type of channels (e.g. EEG or MEG)

trial = {'Data': F, 'Time': time, 'channelNames': channelNames, 'channelPositions': channelPositions, 'channelTypes': channelTypes}


print(1)


















