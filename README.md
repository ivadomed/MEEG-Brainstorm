# MEEG-Brainstorm

Repository for training MEEG datasets with the [ivadomed](https://ivadomed.org) framework in Brainstorm.

# General description of the project
[Brainstorm](https://neuroimage.usc.edu/brainstorm/Introduction) is a widely used analysis software for MEG/EEG/fNIRS/ECoG/Invasive neurophysiology. Brainstorm allows users to perform their entire analysis (importing of raw signals, preprocessing, analysis pipelines, visualizations, create paper figures etc.) without any programming skills needed. Everything is done with the usage of a user-friendly GUI.

The scope of this project is to create seemless interoperability between Brainstorm and Ivadomed.

For this project, we are using a development version of Brainstorm (https://github.com/mpompolas/brainstorm3/tree/ivadomed_repo).
All changes are happening on the Brainstorm side, with the use of converters that create fake "BIDS" MRI datasets that are used as an input to Ivadomed. The term "fake" here refers just to the naming conventions that are used in the dataset and the fact that it is an MRI BIDS, not EEG or MEG BIDS. The dataset is comprised of NIFTI files.

Changes on the Ivadomed side have been kept to the minimum. Users only need to deactivate the BIDS validator, to avoid issues with the naming conventions used:
https://github.com/ivadomed/ivadomed/blob/master/ivadomed/loader/bids_dataframe.py#L105 (if the users install Ivadomed from Brainstorm's plugin manager, this step will be done automatically).
(Change to:  `indexer = pybids.BIDSLayoutIndexer(force_index=force_index, validate=False)`)


The general pipeline can be visualized by the following schematic:
![image](https://user-images.githubusercontent.com/23224563/144139372-d0592453-7f04-4ad7-a59e-ac5301f28757.png)

1. Brainstorm users import and preprocess their EEG/MEG data as usual. The development version of Brainstorm that is linked above, enables a new entry (`Ivadomed Toolbox -> Create BIDS dataset`) in the Brainstorm processing list that allows users to take trials (see definition later in this readme), and convert them to a fake "BIDS" dataset. The created dataset is saved within the brainstorm_tmp folder (find it within `Matlab` by running the command: `bst_get('BrainstormTmpDir')`). 
NOTE: The tmp folder will be emptied if you continue using Brainstorm after the dataset is created.
2. The created dataset is used as an input to ivadomed. Typically, users have to copy the created dataset to a server with GPU power, and run the training on the server side. In case the machine that Brainstorm is installed has reasonable GPU power, the training can be run locally.
Regarding the config file that contains the parameters, the BIDS dataset contains a template `config.json` file that is partially altered to account for entries that are unique to the trials converted (mostly naming conventions: input/output folders, annotation suffix, training/testing suffix etc.).
Part of this project is also to optimize the parameters of the template `config.json` file.
3. Once training is done, users can use the trained model on new datasets. The function (`Ivadomed Toolbox -> Segmentation`) allows users to select the trained model of their choice, and perform segmentation on a new set of trials. The trials are again converted to BIDS dataset and the segmentation is performed through a call to ivadomed. The segmented masks are converted to events that are automatically imported to Brainstorm. e.g. epileptic spikes are annotated as events in time and shown on the top of the signals.


## Additional information

A collection of ivadomed's config.json files will be listed in this repo for different training approaches. Discussions about training strategies will happen:
- In this [google doc](https://docs.google.com/document/d/1PLo__1w8K5Zk1c8ckOLamadaGM_dK872cYwsGFO4DQk/edit#)
- In NeuroPoly/NeuroSPEED slack channel (internal to the dev team)
- In this repository as new [issues](https://github.com/ivadomed/MEEG-Brainstorm/issues).



## Extraction of data using Brainstorm

In order to keep the two software as separate as possible and avoid duplications, all the EEG/MEG data processing is done on the Brainstorm side, and only once everything is prepared it gets transfered to ivadomed. The trials are created after preprocessing (ICA/PCA artifact rejection, resampling, z-scoring, filtering etc.).

The function `Ivadomed Toolbox -> Create BIDS dataset` has a variety of input parameters that affect the dataset created. The following figure shows the pop-up window where the users select the dataset parameters:


![image](https://user-images.githubusercontent.com/23224563/144146427-e11d1789-67de-421b-88d1-57b95e26a870.png)

- `Modality selection`: some acquisition systems contain more than one recording modality in the same file. This selection allows the dataset to select the desired modality or a combination of modalities.
- `Event for ground truth`: if the selected event is present in the trial: in case of a simple event, the `Annotations Time window` will create an annotation around the selected event. In case of an extended event, the annotation time window values are ignored
- `Resampling rate`: if a value is entered here, the signals will be resampled to match this sampling rate. This ultimately affects the 3rd dimension of the NIFTI files.
- `Jitter value`: This values chops a random number of ms, up to the value selected, from both edges of the trials.
- `Gaussian annotation`: If selected, the annotation on the BIDS derivatives will be comprised of a gaussian function (soft annotation) with the 95% edges at the edges of the `Annotations Time window` that is selected around the `Event for ground truth`. `Soft seg training needed on the config file.`
- `Soft annotation threshold`: If selected, the values on the annotation below the threshold will be set to 0, and above the threshold will be left as is. `Soft seg training needed on the config file.`
- `BIDS folders creation`: Since we only have a single subject on this first epilepsy dataset, we fake subjects by selecting: `Separate each trial as different subjects`.
- `FSLeyes`: if `FSLEyes` is installed, it shows an image and its derivative for quality control.


## Definitions

- Trial: Exerpt from an entire recording of an EEG and/or MEG experiment, across channels. Eg: (64 ch, 200 samples).
         Typically trials are created around an event of interest (button press, stimulus presentation etc.)
- Simple event: Event that is just a mark in time - single timepoint.
- Extended event: EVent that is a period of time, not just a timepoint


## Example datasets

Example datasets can be found at NeuroPoly's internal server: duke:temp/konstantinos


## Help needed

Help is needed on the training of the datasets to optimize parameters.
