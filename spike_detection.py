#!/usr/bin/env python

"""
This script is used to train and test the detection model.
It detects spikes in the EEG/MEG trials.
The output of the model is a tensor of logits of dimension
[batch_size x n_time_windows x 2].
The inference is a tensor of dimension [batch_size x n_time_windows]
with 1 in time window w if a spike occurs in it and 0 otherwise.

Usage: type "from spike_detection import <class>" to use class.

Contributors: Ambroise Odonnat.
"""

import json
import os
import torch
import warnings

import numpy as np

from datetime import datetime
from loguru import logger
from sklearn.model_selection import RepeatedKFold, train_test_split
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from custom_losses import get_detection_loss
from common_spatial_pattern import common_spatial_pattern
from data import Data
from dataloader import get_dataloader, get_pad_dataloader
from early_stopping import EarlyStopping
from learning_rate_warmup import NoamOpt
from models import DetectionBertMEEG
from parser import get_parser
from utils import define_device, get_spike_windows, reset_weights

warnings.filterwarnings("ignore")


class DetectionTransformer():

    def __init__(self, config):

        """
        Args:
            config (dict): Dictionnary containing information on data.
        """

        # Data format
        self.Float = torch.FloatTensor
        self.Long = torch.LongTensor

        # Recover data config
        data_config = config['loader_parameters']
        path_root = data_config['path_root']
        wanted_event_label = data_config['wanted_event_label']
        wanted_channel_type = data_config['wanted_channel_type']
        sample_frequence = data_config['sample_frequence']
        binary_classification = data_config['binary_classification']

        # Recover datasets
        logger.info("Get dataset")
        self.dataset = Data(path_root, wanted_event_label,
                            wanted_channel_type, sample_frequence,
                            binary_classification)
        datasets = self.dataset.all_datasets()
        self.all_data, self.all_labels, self.all_spike_events = datasets

    def cross_validation(self, config, save, path_output, gpu_id):

        """ Train and evaluate a detection model using
            a repeated k-fold cross-validation.
            If save is True, the best model parameters and the training and
            validation information are saved to path_output.

        Args:
            config (dict): Dictionnary containing information
                           on data and training configurations.
            save (bool): If True, save best model parameters and training
                         and validation information to path_output.
            path_output (str): Saving path.
            gpu_id (int): Id of the cuda device to use if available.

        Returns:
            results (dict): Contains performance of the model.
        """

        intra_subject = config['intra_subject']
        period = config['period']

        # Recover configurations
        training_config = config['training_parameters']
        model_config = config['model']
        optimizer_config = config['optimizer']
        cross_validation_config = config['cross_validation']

        # Recover training parameters
        batch_size = training_config['batch_size']
        num_workers = training_config['num_workers']
        n_epochs = training_config['epochs']
        mix_up, BETA = training_config['use_mix_up'], training_config['BETA']
        cost_sensitive = training_config['use_cost_sensitive']
        lambd = training_config['lambda']
        l1_penality = training_config['l1_penality']

        # Recover optimizer parameters
        lr = optimizer_config['lr']
        b1 = optimizer_config['b1']
        b2 = optimizer_config['b2']
        weight_decay = optimizer_config['weight_decay']
        amsgrad = optimizer_config['use_amsgrad']
        warmup = optimizer_config['learning_rate_warmup']
        scheduler_config = optimizer_config['scheduler']
        scheduler = scheduler_config['use_scheduler']
        scheduler_patience = scheduler_config['patience']
        factor, min_lr = scheduler_config['factor'], scheduler_config['min_lr']
        el_stop_config = optimizer_config['early_stopping']
        el_stop_patience = el_stop_config['patience']

        # Define training, validation and regularization losses
        self.train_criterion = get_detection_loss(torch.nn.BCEWithLogitsLoss(),
                                                  cost_sensitive, lambd)
        self.val_criterion = torch.nn.BCEWithLogitsLoss()
        self.l1 = torch.nn.L1Loss()
        available, device = define_device(gpu_id)
        if available:
            self.train_criterion = self.train_criterion.cuda()
            self.val_criterion = self.val_criterion.cuda()
            self.l1 = self.l1.cuda()

        # Keep track of results and model parameters
        results = {}
        models = {}
        z_scores = {}
        best_F1 = 0
        best_fold = 0

        # Recover cross_validation parameters
        n_splits = cross_validation_config['n_splits']
        n_repeats = cross_validation_config['n_repeats']
        random_state = cross_validation_config['random_state']

        # Initialize RepeatedKFold
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                            random_state=random_state)

        print('--------------------------------')

        # Recover subjects id
        self.subject_ids = np.asarray(list(self.all_data.keys()))

        # Perform intra-subject learning
        if intra_subject:
            subject_id = self.subject_ids[0]
            logger.info('Spike detection for subject {}.'.format(subject_id))
            all_data = self.all_data[subject_id][0]
            spike_events = self.all_spike_events[subject_id][0]
            index = [i for i in range(all_data.shape[0])]

        # Perform cross-subject learning
        else:
            index = range(self.subject_ids.shape[0])
            np.random.seed(random_state)
            train_index = {}
            val_index = {}
            test_index = {}

        # Loop on the folds
        for fold, (train_ids, test_ids) in enumerate(rkf.split(index)):

            print('FOLD {}'.format(fold))
            print('--------------------------------')
            validation_possible = True
            if intra_subject:

                # Select a validation set to do early stopping
                val_size = cross_validation_config['val_size']
                train_ids, val_ids = train_test_split(train_ids,
                                                      test_size=val_size)

                # Recover training dataset
                train_data = all_data[train_ids]
                train_spike_events = spike_events[train_ids]

                # Z-score standardization
                target_mean = np.mean(train_data)
                target_std = np.std(train_data)
                z_scores[fold] = {'mean': target_mean,
                                  'std': target_std}
                train_data = (train_data-target_mean) / target_std

                # Create training dataloader
                train_data = np.expand_dims(train_data, axis=1)
                train_dataloader = get_dataloader(train_data,
                                                  train_spike_events,
                                                  batch_size, True,
                                                  num_workers)

                # Recover val dataset
                val_data = all_data[val_ids]
                val_spike_events = spike_events[val_ids]

                # Z-score standardization
                val_data = (val_data-target_mean) / target_std

                # Create val dataloader
                val_data = np.expand_dims(val_data, axis=1)
                val_dataloader = get_dataloader(val_data,
                                                val_spike_events,
                                                batch_size, False,
                                                num_workers)
                # Recover test dataset
                test_data = all_data[test_ids]
                test_spike_events = spike_events[test_ids]

                # Z-score standardization
                test_data = (test_data-target_mean) / target_std

                # Create test dataloader
                test_data = np.expand_dims(test_data, axis=1)
                test_dataloader = get_dataloader(test_data,
                                                 test_spike_events,
                                                 batch_size, False,
                                                 num_workers)
            else:

                # If possible, select a validation subject to do early stopping
                validation_possible = len(train_ids) > 1
                if validation_possible:
                    index = np.random.randint(low=0, high=len(train_ids))
                    val_ids = np.array([train_ids[index]])
                    train_ids = np.delete(train_ids, index)
                    val_index[fold] = str(self.subject_ids[val_ids])
                else:
                    val_index[fold] = 'None'
                train_index[fold] = str(train_ids)
                test_index[fold] = str(self.subject_ids[test_ids])

                print('Validation on: {}, '
                      'test on: {}'.format(val_index[fold],
                                           test_index[fold]))

                # Recover training dataset
                train_subject_ids = self.subject_ids[train_ids]
                train_data = []
                for id in train_subject_ids:
                    sessions_trials = self.all_data[id]
                    for trials in sessions_trials:
                        train_data.append(trials)
                train_spike_events = []
                for id in train_subject_ids:
                    sessions_events = self.all_spike_events[id]
                    for events in sessions_events:
                        train_spike_events.append(events)

                # Z-score standardization
                target_mean = np.mean([np.mean(data) for data in train_data])
                target_std = np.mean([np.std(data) for data in train_data])
                z_scores[fold] = {'mean': target_mean,
                                  'std': target_std}
                train_data = [np.expand_dims((data-target_mean) / target_std,
                                             axis=1)
                              for data in train_data]

                # Create training dataloader
                train_dataloader = get_pad_dataloader(train_data,
                                                      train_spike_events,
                                                      batch_size, True,
                                                      num_workers)
                if validation_possible:

                    # Recover val dataset
                    val_subject_ids = self.subject_ids[val_ids]
                    val_data = []
                    for id in val_subject_ids:
                        sessions_trials = self.all_data[id]
                        for trials in sessions_trials:
                            val_data.append(trials)
                    val_spike_events = []
                    for id in val_subject_ids:
                        sessions_events = self.all_spike_events[id]
                        for events in sessions_events:
                            val_spike_events.append(events)

                    # Z-score standardization
                    val_data = [np.expand_dims((data-target_mean) / target_std,
                                               axis=1)
                                for data in val_data]

                    # Create val dataloader
                    val_dataloader = get_pad_dataloader(val_data,
                                                        val_spike_events,
                                                        batch_size, False,
                                                        num_workers)

                # Recover test dataset
                test_subject_ids = self.subject_ids[test_ids]
                test_data = []
                for id in test_subject_ids:
                    sessions_trials = self.all_data[id]
                    for trials in sessions_trials:
                        test_data.append(trials)
                test_spike_events = []
                for id in test_subject_ids:
                    sessions_events = self.all_spike_events[id]
                    for events in sessions_events:
                        test_spike_events.append(events)

                # Z-score standardization
                test_data = [np.expand_dims((data-target_mean) / target_std,
                                            axis=1)
                             for data in test_data]

                # Create test dataloader
                test_dataloader = get_pad_dataloader(test_data,
                                                     test_spike_events,
                                                     batch_size, False,
                                                     num_workers)

            # Initialize detection neural network
            model = DetectionBertMEEG(**model_config)
            model.apply(reset_weights)

            # Move to gpu if available
            if available:
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)
            model.to(device)

            # Define optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                         betas=(b1, b2),
                                         weight_decay=weight_decay,
                                         amsgrad=amsgrad)

            # Define warmup method
            if warmup:
                warmup_scheduler = NoamOpt(lr, warmup, optimizer)

            # Define scheduler
            lr_scheduler = ReduceLROnPlateau(optimizer, 'min',
                                             factor=factor,
                                             patience=scheduler_patience,
                                             min_lr=min_lr, verbose=False)

            # Define early stopping
            early_stopping = EarlyStopping(patience=el_stop_patience)

            # Loop over the dataset n_epochs times
            for e in range(n_epochs):

                # Train and validate the model
                train_loss, train_steps = 0, 0
                val_loss, val_steps = 0, 0

                # Set training mode
                model.train()
                for i, (data, labels) in enumerate(train_dataloader):

                    # Apply time window reduction
                    n_time_windows = model_config['n_time_windows']
                    labels = get_spike_windows(labels, n_time_windows)
                    if mix_up:

                        """
                        Apply a mix-up strategy for data augmentation.
                        Inspired by:
                        `"mixup: Beyond Empirical Risk Minimization"
                        <https://arxiv.org/abs/1710.09412>`_.
                        """

                        # Roll a copy of the batch
                        rl_factor = torch.randint(0, data.shape[0],
                                                  (1,)).item()
                        rl_data = torch.roll(data, rl_factor, dims=0)
                        rl_labels = torch.roll(labels,
                                               rl_factor, dims=0)

                        # Create a tensor of lambdas sampled
                        # from the beta distribution
                        lambdas = np.random.beta(BETA, BETA, data.shape[0])

                        # Trick inspired by:
                        # `<https://forums.fast.ai/t/mixup-data-augmentation/22764>`
                        lambdas = torch.reshape(torch.tensor
                                                (np.maximum(lambdas,
                                                            1-lambdas)),
                                                (-1, 1, 1, 1))

                        # Mix samples
                        mix_data = lambdas*data + (1-lambdas)*rl_data

                        # Format conversion and move to device
                        mix_data = Variable(mix_data.type(self.Float),
                                            requires_grad=True)
                        data = Variable(data.type(self.Float),
                                        requires_grad=True)
                        labels = Variable(labels.type(self.Float))
                        rl_labels = Variable(rl_labels.type(self.Float))
                        mix_data = mix_data.to(device)
                        labels = labels.to(device)
                        rl_labels = rl_labels.to(device)

                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # Forward
                        _, mix_outputs = model(mix_data)

                        # Compute mix-up loss
                        lambdas = lambdas.squeeze().to(device)
                        loss1 = self.train_criterion(mix_outputs,
                                                     labels)
                        loss2 = self.train_criterion(mix_outputs,
                                                     rl_labels)
                        loss = lambdas*loss1 + (1-lambdas)*loss2
                        loss = loss.sum()

                        # Apply L1 regularization
                        l1_loss = 0
                        for param in model.parameters():
                            zeros = torch.zeros(param.size()).to(device)
                            l1_loss += self.l1(param, zeros)
                        loss += l1_penality*l1_loss

                        # Backward
                        loss.backward()

                        # Update learning rate
                        if warmup:
                            warmup_scheduler.step()
                        else:
                            optimizer.step()

                        # Detach from device
                        loss = loss.cpu().detach().numpy()

                        # Recover training loss
                        train_loss += loss
                        train_steps += 1
                    else:

                        # Format conversion and move to device
                        data = Variable(data.type(self.Float),
                                        requires_grad=True)
                        labels = Variable(labels.type(self.Float))
                        data, labels = data.to(device), labels.to(device)

                        # Zero the parameter gradients
                        optimizer.zero_grad()

                        # Forward
                        _, outputs = model(data)

                        # Compute training loss
                        loss = self.train_criterion(outputs,
                                                    labels)

                        # Apply L1 regularization
                        l1_loss = 0
                        for param in model.parameters():
                            zeros = torch.zeros(param.size()).to(device)
                            l1_loss += self.l1(param, zeros)
                        loss += l1_penality*l1_loss

                        # Backward
                        loss.backward()

                        # Update learning rate
                        if warmup:
                            warmup_scheduler.step()
                        else:
                            optimizer.step()

                        # Detach from device
                        loss = loss.cpu().detach().numpy()

                        # Recover training loss
                        train_loss += loss
                        train_steps += 1

                # Print loss
                train_loss /= train_steps
                if (e+1) % period == 0:
                    logger.info('Training loss '
                                'at epoch {}: {}'.format(e+1, train_loss))
                if validation_possible:

                    # Set evaluation mode
                    model.eval()
                    for j, (data, labels) in enumerate(val_dataloader):

                        # Apply time window reduction
                        n_time_windows = model_config['n_time_windows']
                        labels = get_spike_windows(labels, n_time_windows)

                        # Format conversion and move to device
                        data = Variable(data.type(self.Float),
                                        requires_grad=True)
                        labels = Variable(labels.type(self.Float))
                        data = data.to(device)
                        labels = labels.to(device)

                        # Forward
                        _, outputs = model(data)

                        # Compute training loss
                        loss = self.val_criterion(outputs, labels)

                        # Detach from device
                        loss = loss.cpu().detach().numpy()

                        # Recover training loss and confusion matrix
                        val_loss += loss
                        val_steps += 1

                    # Print loss
                    val_loss /= val_steps
                    if (e+1) % period == 0:
                        logger.info('Validation loss at epoch '
                                    '{}: {}'.format(e+1, val_loss))

                    # Update learning rate if training loss
                    # does not improve
                    if scheduler:
                        lr_scheduler.step(val_loss)

                    # Update early stopping
                    early_stopping(val_loss)
                    if early_stopping.early_stop:
                        final_epoch = e+1
                        print("Early stopping at epoch: ", final_epoch)
                        break

            # Training finished
            logger.info('Training process finished.')

            # Recover model parameters
            models[fold] = model.state_dict()

            # Print about testing
            logger.info('Start evaluation.')

            # Set evaluation mode
            model.eval()

            # Initialize confusion_matrix
            # Here the positive class is the class 1
            confusion_matrix = np.zeros((2, 2))

            with torch.no_grad():
                for k, (data, labels) in enumerate(test_dataloader):

                    # Apply time window reduction
                    n_time_windows = model_config['n_time_windows']
                    labels = get_spike_windows(labels, n_time_windows)

                    # Format conversion and move to device
                    data = Variable(data.type(self.Float),
                                    requires_grad=True)
                    labels = Variable(labels.type(self.Long))
                    data, labels = data.to(device), labels.to(device)

                    # Inference
                    _, outputs = model(data)
                    pred = (outputs > 0.5).int()

                    # Detach from device
                    labels = labels.cpu().detach()
                    pred = pred.cpu().detach()

                    # Recover confusion matrix
                    for t, p in zip(labels.reshape(-1), pred.reshape(-1)):
                        confusion_matrix[t.long(), p.long()] += 1

                logger.info('Evaluation finished.\n')

                TP = confusion_matrix[1][1]
                TN = confusion_matrix[0][0]
                FP = confusion_matrix[0][1]
                FN = confusion_matrix[1][0]

                P = TP + FN
                N = TN + FP

                Accuracy = (TP+TN) / (P+N)
                Specificity = TN/N
                Sensitivity = TP/P
                Precision = TP / (TP+FP)
                F1_score = 2 * ((Sensitivity*Precision)
                                / (Sensitivity+Precision))

                # When no spike occurs
                if (TP == 0) & (FP == 0) & (FN == 0):
                    Sensitivity = 1
                    Precision = 1
                    F1_score = 1
                elif (TP == 0) & (FP == 0) & (FN != 0):
                    Sensitivity = 0
                    Precision = 1
                    F1_score = 0
                elif (TP == 0) & (FP != 0) & (FN == 0):
                    Sensitivity = 1
                    Precision = 0
                    F1_score = 0
                elif (Sensitivity == 0) & (Precision == 0):
                    F1_score = 0

                # Update best F1_score
                if F1_score > best_F1:
                    best_fold = fold
                    best_F1 = F1_score

                # Print accuracy
                print('Accuracy for fold {}: {}'.format(fold,
                                                        Accuracy))
                print('Specificity for fold {}: {}'.format(fold,
                                                           Specificity))
                print('Sensitivity for fold {}: {}'.format(fold,
                                                           Sensitivity))
                print('Precision for fold {}: {}'.format(fold,
                                                         Precision))
                print('F1_score for fold {}: {}'.format(fold,
                                                        F1_score))
                print('--------------------------------')

                if intra_subject:

                    # Recover metrics for fold
                    results[fold] = {'Accuracy': Accuracy,
                                     'Specificity': Specificity,
                                     'Sensitivity': Sensitivity,
                                     'Precision': Precision,
                                     'F1_score': F1_score}
                else:
                    # Recover metrics for fold
                    subject = test_index[fold]
                    results[subject] = {'Accuracy': Accuracy,
                                        'Specificity': Specificity,
                                        'Sensitivity': Sensitivity,
                                        'Precision': Precision,
                                        'F1_score': F1_score}
        if intra_subject:

            # Print fold results
            print('INTRA-SUBJECT K-FOLD CROSS VALIDATION RESULTS'
                  ' FOR {} FOLDS'.format(n_splits*n_repeats))
            print('--------------------------------')
            avg_acc = np.mean([results[fold]['Accuracy']
                               for fold in range(n_splits*n_repeats)])
            avg_spec = np.mean([results[fold]['Specificity']
                                for fold in range(n_splits*n_repeats)])
            avg_sens = np.mean([results[fold]['Sensitivity']
                                for fold in range(n_splits*n_repeats)])
            avg_prec = np.mean([results[fold]['Precision']
                                for fold in range(n_splits*n_repeats)])
            avg_F1 = np.mean([results[fold]['F1_score']
                              for fold in range(n_splits*n_repeats)])
            results['Mean'] = {'Accuracy': avg_acc,
                               'Specificity': avg_spec,
                               'Sensitivity': avg_sens,
                               'Precision': avg_prec,
                               'F1_score': avg_F1}
            print('Average results: {}\n'.format(results['Mean']))

            # Saving the best model
            print('Best F1 score for fold {}.\n'.format(best_fold))
            if save:

                # Create unique folder ID based on time
                eventid = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                data_config = config['loader_parameters']
                if data_config['wanted_channel_type'] == ['EEG']:
                    path = path_output + 'EEG_spike_detection_attempt_'
                    path += eventid + '/'
                elif data_config['wanted_channel_type'] == ['MEG']:
                    path = path_output + 'MEG_spike_detection_attempt_'
                    path += eventid + '/'
                else:
                    path = path_output + 'MEEG_spike_detection_attempt_'
                    path += eventid + '/'
                try:
                    os.mkdir(path)
                except OSError:
                    logger.info('Creation of the directory %s failed' % path)
                else:
                    logger.info('Successfully created folder %s \n' % path)

                # Save information
                model_path = path + 'model_' + subject_id + '.pth'
                torch.save(models[best_fold], model_path)

                logger.info('Saving results.\n')
                results_path = path + 'results_' + subject_id + '.json'
                json.dump(results, open(results_path, 'w'), indent=4)

                logger.info('Saving configuration file.')
                config_path = path + 'config_' + subject_id + '.json'
                config['save'] = {'output': path_output,
                                  'model': model_path,
                                  'results': results_path,
                                  'config': config_path}
                config['z_score'] = z_scores[best_fold]
                json.dump(config, open(config_path, 'w'), indent=4)
                logger.info('Information saved in {}'.format(path))

            return results
        else:

            # Print fold results
            print('CROSS-SUBJECT K-FOLD CROSS VALIDATION RESULTS'
                  ' FOR {} FOLDS'.format(n_splits*n_repeats))
            print('--------------------------------')
            avg_acc = np.mean([results[test_index[fold]]['Accuracy']
                               for fold in range(n_splits*n_repeats)])
            avg_spec = np.mean([results[test_index[fold]]['Specificity']
                                for fold in range(n_splits*n_repeats)])
            avg_sens = np.mean([results[test_index[fold]]['Sensitivity']
                                for fold in range(n_splits*n_repeats)])
            avg_prec = np.mean([results[test_index[fold]]['Precision']
                                for fold in range(n_splits*n_repeats)])
            avg_F1 = np.mean([results[test_index[fold]]['F1_score']
                              for fold in range(n_splits*n_repeats)])
            results['Mean'] = {'Accuracy': avg_acc,
                               'Specificity': avg_spec,
                               'Sensitivity': avg_sens,
                               'Precision': avg_prec,
                               'F1_score': avg_F1}
            print('Average results: {}\n'.format(results['Mean']))

            # Saving the best model
            print('Best F1 score on fold {} '
                  'for subject: '
                  '{}.\n'.format(best_fold, test_index[best_fold]))
            if save:

                # Create unique folder ID based on time
                eventid = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                data_config = config['loader_parameters']
                if data_config['wanted_channel_type'] == ['EEG']:
                    path = path_output + 'EEG_spike_detection_attempt_'
                    path += eventid + '/'
                elif data_config['wanted_channel_type'] == ['MEG']:
                    path = path_output + 'MEG_spike_detection_attempt_'
                    path += eventid + '/'
                else:
                    path = path_output + 'MEEG_spike_detection_attempt_'
                    path += eventid + '/'
                try:
                    os.mkdir(path)
                except OSError:
                    logger.info('Creation of the directory %s failed' % path)
                else:
                    logger.info('Successfully created folder %s \n' % path)

                logger.info('Saving model.\n')
                model_path = path + 'model.pth'
                torch.save(models[best_fold], model_path)

                logger.info('Saving results.\n')
                results_path = path + 'results.json'
                json.dump(results, open(results_path, 'w'), indent=4)

                logger.info('Saving configuration file.')
                config_path = path + 'config.json'
                config['save'] = {'output': path,
                                  'model': model_path,
                                  'results': results_path,
                                  'config': config_path}
                config['z_score'] = z_scores[best_fold]
                config['split'] = {'validation': val_index[best_fold],
                                   'test': test_index[best_fold]}
                json.dump(config, open(config_path, 'w'), indent=4)
                logger.info('Information saved in {}'.format(path))

            return results


def evaluate(path_root, config, save, path_output, gpu_id):

    """ Detect epilepetic spikes on a given dataset.

    Args:
        path_root (str): Path to the dataset.
        config (dict): Configuration file.
        save (bool): If True, save the results and predictions.
        path_output (str): Path to save the model's performances
                           and predictions.
        gpu_id (int): Id of the cuda device to use if available.

    Return:
        results (dict): Contains performances of the model.
        all_events (dict): Contains the start and end point
                           of predicted spike events.
    """
    # Data format
    Float = torch.FloatTensor
    Long = torch.LongTensor

    # Recover data configuration
    data_config = config['loader_parameters']
    wanted_event_label = data_config['wanted_event_label']
    wanted_channel_type = data_config['wanted_channel_type']
    sample_frequence = data_config['sample_frequence']
    binary_classification = data_config['binary_classification']

    # Recover datasets
    logger.info("Get dataset")
    dataset = Data(path_root, wanted_event_label, wanted_channel_type,
                   sample_frequence, binary_classification)
    datasets = dataset.all_datasets()
    all_data, all_labels, all_spike_events = datasets

    # Recover configurations
    training_config = config['training_parameters']
    model_config = config['model']
    save_config = config['save']

    # Recover dataloader parameters
    batch_size = training_config['batch_size']
    num_workers = training_config['num_workers']

    available, device = define_device(gpu_id)

    # Recover predictions
    all_events = {}

    # Recover subjects id
    subject_ids = np.asarray(list(all_data.keys()))

    intra_subject = config['intra_subject']

    # Perform intra-subject learning
    if intra_subject:
        subject_id = subject_ids[0]
        logger.info('Spike detection for subject {}.'.format(subject_id))
        all_data = all_data[subject_id]
        all_spike_events = all_spike_events[subject_id]

        # Z-score standardization
        target_mean = config['z_score']['mean']
        target_std = config['z_score']['std']
        all_data = (all_data-target_mean) / target_std

        # Apply spatial filter if possible
        n_rows = all_data.shape[1]
        selected_rows = config['selected_rows']
        if n_rows > 2*selected_rows:
            logger.info(' Common spatial pattern algorithm is applied')
            filter_path = save_config['spatial_filter']
            _ = filter_path.seek(0)
            spatial_filter = np.load(filter_path)
            all_data = np.einsum('c d, n d t -> n c t',
                                 spatial_filter, all_data)

        # Create dataloader
        all_data = np.expand_dims(all_data, axis=1)
        dataloader = get_dataloader(all_data,
                                    all_spike_events,
                                    batch_size, False,
                                    num_workers)

    # Perform cross-subject learning
    else:

        # Recover training dataset
        all_data = [all_data[id] for id in subject_ids]
        all_spike_events = [all_spike_events[id]
                            for id in subject_ids]

        # Z-score standardization
        target_mean = config['z_score']['mean']
        target_std = config['z_score']['std']
        all_data = (all_data-target_mean) / target_std
        all_data = [np.expand_dims((data-target_mean) / target_std, axis=1)
                    for data in all_data]

        # Create training dataloader
        dataloader = get_pad_dataloader(all_data,
                                        all_spike_events,
                                        batch_size, False,
                                        num_workers)

    # Recover detection neural network
    path_model = save_config['model']
    model = DetectionBertMEEG(**model_config)
    model.load_state_dict(torch.load(path_model))

    # Move to gpu if available
    if available:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    model.to(device)

    logger.info('Start evaluation.')

    # Set evaluation mode
    model.eval()

    # Recover time windows
    n_time_points = model_config['n_time_points']
    n_time_windows = model_config['n_time_windows']
    time = torch.Tensor([i/sample_frequence for i in range(n_time_points)])
    time_windows = torch.chunk(time, n_time_windows)

    # Initialize confusion_matrix
    # Here the positive class is the class 1
    confusion_matrix = np.zeros((2, 2))

    with torch.no_grad():
        for i, (data, labels) in enumerate(dataloader):

            # Apply time window reduction
            labels = get_spike_windows(labels, n_time_windows)

            # Format conversion and move to device
            data = Variable(data.type(Float),
                            requires_grad=True)
            labels = Variable(labels.type(Long))
            data, labels = data.to(device), labels.to(device)

            # Forward
            _, outputs = model(data)
            pred = (outputs > 0.5).int()

            # Detach from device
            labels = labels.cpu().detach()
            pred = pred.cpu().detach()

            # Recover predicted spike events start and end times
            events = {}
            for b in range(pred.size(0)):
                index = (pred[b] == 1).int().nonzero().numpy()
                w = []
                for i in index:
                    i = i[0]
                    time_window = time_windows[i].numpy()
                    start, end = time_window[0], time_window[-1]
                    w.append((start, end))
                events[b] = w
            all_events[i] = events

            # Recover confusion matrix
            for t, p in zip(labels.reshape(-1), pred.reshape(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        logger.info('Evaluation finished.\n')

        TP = confusion_matrix[1][1]
        TN = confusion_matrix[0][0]
        FP = confusion_matrix[0][1]
        FN = confusion_matrix[1][0]

        P = TP + FN
        N = TN + FP

        Accuracy = (TP+TN) / (P+N)
        Specificity = TN/N
        Sensitivity = TP/P
        Precision = TP / (TP+FP)
        F1_score = 2 * ((Sensitivity*Precision)
                        / (Sensitivity+Precision))

        # Print accuracy
        print('Accuracy  {}: '.format(Accuracy))
        print('Specificity {}: '.format(Specificity))
        print('Sensitivity {}: '.format(Sensitivity))
        print('Precision {}:'.format(Precision))
        print('F1_score {}:'.format(F1_score))
        print('--------------------------------')

        # Recover metrics for fold
        results = {'Accuracy': Accuracy,
                   'Specificity': Specificity,
                   'Sensitivity': Sensitivity,
                   'Precision': Precision,
                   'F1_score': F1_score}

        if save:
            logger.info('Saving results.\n')
            results_path = path_output + 'results.json'
            json.dump(results, open(results_path, 'w'), indent=4)

            logger.info('Saving predicted events.\n')
            events_path = path_output + 'predicted_events.json'
            json.dump(all_events, open(events_path, 'w'), indent=4)

            logger.info('Information saved in {}'.format(path_output))

        return results, all_events


def main():

    """ Train or evaluate DetectionBertMEEG.

        Training: run the following command in terminal to perform
                  cross-validation and save the best model.

                python spike_detection.py --train --save
                --path-root [path to data]
                --path-config [path to config file]
                --path-output [path to output folder] --gpu_id 0

            Example of config file:

            config = {
                    'loader_parameters': {
                        'path_root': '../EEG_signals/',
                        'wanted_event_label': 'saw_EST',
                        'wanted_channel_type': ['EEG'] ,
                        'sample_frequence': 100,
                        'binary_classification': True
                        },
                    'intra_subject': True,
                    'selected_rows': 10,
                    'training_parameters': {
                        'batch_size': 8,
                        'num_workers': 0,
                        'Epochs': 100,
                        'Mix-up': True,
                        'BETA': 0.4,
                        'use_cost_sensitive': True,
                        'lambda': 10,
                        },
                    'model': {
                        'n_time_points': 201,
                        'attention_num_heads': 3,
                        'attention_dropout': 0.8,
                        'attention_kernel': 30,
                        'attention_stride': 5,
                        'spatial_dropout': 0.8,
                        'emb_size': 30,
                        'n_maps': 5,
                        'position_kernel': 20,
                        'position_stride': 1,
                        'channels_kernel': 20,
                        'channels_stride': 1,
                        'time_kernel': 20,
                        'time_stride': 1,
                        'embedding_dropout': 0,
                        'depth': 3,
                        'num_heads': 10,
                        'expansion': 4,
                        'transformer_dropout': 0,
                        'n_time_windows': 10,
                        'detector_dropout': 0
                        },
                    'optimizer': {
                        'lr': 1e-3,
                        'b1': 0.9,
                        'b2': 0.999,
                        'weight_decay': 0,
                        'use_amsgrad': False,
                        'learning_rate_warmup': 0,
                        'scheduler': {
                            'use_scheduler': False,
                            'patience': 20,
                            'factor': 0.5,
                            'min_lr': 1e-5
                            }
                        }
                    'cross_validation': {
                        'n_splits': n_splits,
                        'n_repeats': n_repeats,
                        'random_state': random_state
                        }
                    }

        Test: run the following command in terminal to realise inference
              with the best model from cross-validation.

                python spike_detection.py --test --save
                --path-root [path to data]
                --path-config [path to config file]
                --path-output [path to output folder] --gpu_id 0

            Example of config file:

            config = {
                    'loader_parameters': {
                        'path_root': '../EEG_signals/',
                        'wanted_event_label': 'saw_EST',
                        'wanted_channel_type': ['EEG'] ,
                        'sample_frequence': 100,
                        'binary_classification': True
                        },
                    'intra_subject': True,
                    'selected_rows': 10,
                    'training_parameters': {
                        'batch_size': 8,
                        'num_workers': 0,
                        'Epochs': 100,
                        'Mix-up': True,
                        'BETA': 0.4,
                        'use_cost_sensitive': True,
                        'lambda': 10,
                        },
                    'model': {
                        'n_time_points': 201,
                        'attention_num_heads': 3,
                        'attention_dropout': 0.8,
                        'attention_kernel': 30,
                        'attention_stride': 5,
                        'spatial_dropout': 0.8,
                        'emb_size': 30,
                        'n_maps': 5,
                        'position_kernel': 20,
                        'position_stride': 1,
                        'channels_kernel': 20,
                        'channels_stride': 1,
                        'time_kernel': 20,
                        'time_stride': 1,
                        'embedding_dropout': 0,
                        'depth': 3,
                        'num_heads': 10,
                        'expansion': 4,
                        'transformer_dropout': 0,
                        'n_time_windows': 10,
                        'detector_dropout': 0
                        },
                    'optimizer': {
                        'lr': 1e-3,
                        'b1': 0.9,
                        'b2': 0.999,
                        'weight_decay': 0,
                        'use_amsgrad': False,
                        'learning_rate_warmup': 0,
                        'scheduler': {
                            'use_scheduler': False,
                            'patience': 20,
                            'factor': 0.5,
                            'min_lr': 1e-5
                            }
                        }
                    'cross_validation': {
                        'n_splits': n_splits,
                        'n_repeats': n_repeats,
                        'random_state': random_state
                        }
                    'save': {'output': ../EEG_signals/,
                        'model': ../model.pt,
                        'spatial_filter': ../filter.npy,
                        'results': ../results.json,
                        'config': ../config.json
                        },
                    'z_score': {'mean': 0.7,
                        'std': 0.1
                        },
                    'split': {'train': [0, 2, 3, 4],
                        'test': [1, 5]
                        }
                    }
    """

    parser = get_parser()
    args = parser.parse_args()

    # Recover commands
    train = args.train
    test = args.test

    # Recover paths
    path_config = args.path_config
    path_output = args.path_output

    # Recover config dictionnary
    with open(path_config) as f:
        config = json.loads(f.read())

    # Recover save and gpu
    save = args.save
    gpu_id = args.gpu_id

    # Train model
    if train:
        path_root = args.path_root
        config['loader_parameters']['path_root'] = path_root
        detection = DetectionTransformer(config)
        train_results = detection.cross_validation(config, save,
                                                   path_output, gpu_id)

    # Evaluate model
    elif test:
        path_root = args.path_root
        test_results = evaluate(path_root, config, save,
                                path_output, gpu_id)


if __name__ == "__main__":
    main()
