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

![image](https://user-images.githubusercontent.com/23224563/164266867-53d89a44-43f7-4503-bffc-d48d7b1965d7.png)


The function `Ivadomed Toolbox -> Create BIDS dataset` has a variety of input parameters that affect the dataset created. The following figure shows the pop-up window where the users select the dataset parameters:


![image](https://user-images.githubusercontent.com/23224563/164158717-a80f3c5e-67fa-4509-83aa-2d6adcc84ade.png)

- `Parent folder`: Parent folder to store the created BIDS folder. If selected, the BIDS folder included within it will have the following name: 'protocolName_modality_datasetID_#' where # is a sequential number for easy identification. If no parent folder is selected, the BIDS dataset will be created within Brainstorm's temp folder - ATTENTION: the temp folder is emptied every time this function is used, or when Brainstorm is restarted.

**Modality selection**: Some acquisition systems contain more than one recording modality (ie: EEG and/or MEG) in the same file. This selection allows the dataset to select the desired modality or a combination of modalities.

**Annotation parameters**
- `Event for ground truth`: Events to be used as ground truth annotations in the NIfTIs. 
- `Annotations Time window`: In case of a simple event, this field is used to create an annotation around the selected event. In case of an extended event, the annotation time window values are ignored.
- `Whole head annotation:` 
  - `Whole head`: the entire slice is annotated.
  - `Partial`: Only the annotated channels are annotated with the peak of a Gaussian centered at those channels (Gaussian in space).
- `Include trials that don't have selected events`: if a trial does not have a ground truth, include it. In this case, the ground truth (stored under `/derivatives`) will be a whole-zero NIfTI file.
- `Gaussian annotation`: The annotation on the NIfTI file will be smoothed by a Gaussian function across time. The sigma of the Gaussian function is adjusted so that 5% bilateral edges are beyond the `Annotations Time window` defined earlier.

**BIDS conversion parameters**
- `Resampling rate`: The signals will be resampled to match this sampling rate. This ultimately affects the 3rd dimension of the NIFTI files (ie: time).
- `Jitter value`: Range (in ms) within which a random number is selected to crop both edges of the trials. For example: if a trial is 1000ms long, a Jitter value of 100ms will crop both edges of the trial, producing a trial that is between 800ms and 1000ms long.

**Data augmentation parameters (signal level)**
- `Channels drop-out`: remove `n` random channels from the M/EEG dataset. `n` is a random number between 0 and the value input in this field. After removing those channels, the data will be interpolated on the Euclidean grid defined by the NIfTI file. For example, if a value `5` is selected, then up to 5 channels will be randomly removed for each trial independently (eg: trial 1 will have 2 channels removed, trial 2 will have 4 channels removed, etc.). This is used to emulate the fact that recordings in different subjects can sometimes have some channels missing/corrupted.

**BIDS folders creation**
In ivadomed, training imposes to split data across subjects. Hence, if training of M/EEG data is only done in one subject, we need to find a "hack" to split the data into several virtual subjects. This is the purpose of this section.
- `Normal`: Assign trials to the correct subjects. 
- `Separate runs/session as different subjects`: Each run is a different subject.
- `Separate each trial as different subjects`: Each trial is a different subject.

**FSLeyes**
- `Open an example image/derivative on FSLeyes`: If [FSLeyes](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLeyes) is installed, it shows an image and its derivative for quality control.


## Definitions

- Trial: Exerpt from an entire recording of an EEG and/or MEG experiment, across channels. Eg: (64 ch, 200 samples).
         Typically trials are created around an event of interest (button press, stimulus presentation etc.)
- Simple event: Event that is just a mark in time - single timepoint.
- Extended event: EVent that is a period of time, not just a timepoint


## Example datasets

Example datasets can be found at NeuroPoly's internal server: duke:temp/konstantinos


## Help needed

Help is needed on the training of the datasets to optimize parameters.

## Relevant literature

1. [Transformer based model used in ao/seizure_classification](https://arxiv.org/pdf/2106.11170.pdf).

   a. [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
   
2. [Detection of mesial temporal lobe epileptiform discharges on intracranial electrodes using deep learning](https://pubmed.ncbi.nlm.nih.gov/31760212/)
3. [Neurophysiologically interpretable DNN predicts complex movement components from brain activity](https://www.nature.com/articles/s41598-022-05079-0)

