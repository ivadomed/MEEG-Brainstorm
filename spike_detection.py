#!/usr/bin/env python

"""
This script is used to train and test the detection model. It detects spikes in the EEG/MEG trials.
The output of the model is a tensor of logits of dimension [batch_size x n_time_windows x 2].
The inference is a tensor of dimension [batch_size x n_time_windows] 
with value 1 in time window t if a spike occurs in it and 0 otherwise.

Usage: type "from spike_detection import <class>" to use one of its classes.
       
Contributors: Ambroise Odonnat.
"""


import json
import torch
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from einops import rearrange 
from loguru import logger
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import RepeatedKFold
from torch import nn 
from torch.autograd import Variable

from custom_losses import get_detection_loss
from common_spatial_pattern import common_spatial_pattern
from data import Data
from dataloader import get_dataloader, get_cross_subject_dataloader
from early_stopping import EarlyStopping
from learning_rate_warmup import NoamOpt
from models import DetectionBertMEEG
from parser import get_parser
from utils import define_device, get_spike_windows, reset_weights


class DetectionTransformer():
    
    def __init__(self, config):
        
        """    
        Args:
            config (dict): Dictionnary containing information on data, training, model and optimizer configurations.
            intra_subject (bool): If True, apply model to a single subject and split its trials between training, validation and test.
                                  Otherwise, apply model to all subjects available and split subjects into training, validation and test sets.
                                  
        Example of config file:
        
            config = {
                'loader_parameters': {
                    'path_root': '../../Neuropoly_Internship/MEEG_data/EEG_signals/',
                    'wanted_event_label': 'saw_EST',
                    'wanted_channel_type': ['EEG'] ,
                    'binary_classification': True
                    }
                }                                                                    
        """
        
        # Data format
        self.Tensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor

        # Recover data config
        data_config = config['loader_parameters']
        path_root = data_config['path_root']
        wanted_event_label = data_config['wanted_event_label']
        wanted_channel_type = data_config['wanted_channel_type']
        binary_classification = data_config['binary_classification']
        
        # Recover training, validation and testing datasets
        logger.info("Get dataset")
        self.dataset = Data(path_root, wanted_event_label, wanted_channel_type, binary_classification)
        self.all_data, self.all_labels, self.all_spike_events = self.dataset.all_datasets()


    def cross_validation(self, config, save, path_output, gpu_id):

        """
        Train a detection model using a Nxk-fold cross-validation strategy.
        We use an early stopping strategy which stops the training process when the validation loss does not improve anymore.
        If save is True, training and validation informations, model parameters and configuration file are saved
        in path_output for each fold.

        Args:
            config (dict): Dictionnary containing information on data, training, model and optimizer configurations.
            save (bool): If True, save model parameters and config file in path_output folder.
            path_output (str): Folder path to save training and validation informations, model parameters and configuration file.
            gpu_id (int): Id of the cuda device to use if available.

        Returns:
            train_info (dict): Contains loss and metrics on training set.
            test_info (dict): Contains loss and metrics on validation set.
            final_epoch (int): Last epoch of training.
            
        Example of config file:
        
            config = {
                'loader_parameters': {
                    'path_data': '../../Neuropoly_Internship/MEEG_data/EEG_signals/seconds_2_trials/',
                    'path_channel': '../../Neuropoly_Internship/MEEG_data/EEG_signals/channel_ctf_acc1.mat',
                    'wanted_event_label': 'saw_EST',
                    'wanted_channel_type': ['EEG'] ,
                    'binary_classification': True
                    },
                'training_parameters': {
                    'k_folds': 10,
                    'n_cross_validation': 10,
                    'batch_size': 8,
                    'num_workers': 0,
                    'Epochs': 100
                    'Mix-up': True,
                    'BETA': 0.4,
                    'use_cost_sensitive': True,
                    'lambda': 10,
                    'early_stopping': {
                        'patience': 20,
                        'checkpoint': 'detection_EEG_checkpoint.pt'
                        }
                    },
                'model': {
                    'n_channels': 20,
                    'n_time_points': 201,
                    'attention_dropout': 0.8,
                    'attention_kernel': 30,
                    'attention_stride': 5,
                    'spatial_dropout': 0.8,
                    'position_kernel': 20,
                    'position_stride': 1,
                    'emb_size': 30,
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
                } 
        """
        
        # Recover configurations
        training_config , model_config, optimizer_config = config['training_parameters'], config['model'], config['optimizer']
        cross_validation_config = config['cross_validation']
        CSP_projection_config = config['CSP_projection']
        
        # Recover training parameters
        batch_size, num_workers = training_config['batch_size'], training_config['num_workers']
        n_epochs = training_config['epochs']
        mix_up, BETA = training_config['use_mix_up'], training_config['BETA']
        cost_sensitive, lambd = training_config['use_cost_sensitive'], training_config['lambda']
        
        # Recover optimizer parameters
        lr, b1, b2 = optimizer_config['lr'], optimizer_config['b1'], optimizer_config['b2']
        weight_decay = optimizer_config['weight_decay'] 
        amsgrad = optimizer_config['use_amsgrad']
        warmup = optimizer_config['learning_rate_warmup']
        scheduler_config = optimizer_config['scheduler']
        scheduler, scheduler_patience = scheduler_config['use_scheduler'], scheduler_config['patience']
        factor, min_lr = scheduler_config['factor'], scheduler_config['min_lr'] 
        
        # Define training and validation losses
        self.train_criterion_cls = get_detection_loss(cost_sensitive, lambd)
        self.val_criterion_cls = torch.nn.CrossEntropyLoss()

        available, device = define_device(gpu_id)
        if available:
            self.train_criterion_cls = self.train_criterion_cls.cuda()
            self.val_criterion_cls = self.val_criterion_cls.cuda()
        
        # For fold results
        results = {}
          
        # Recover cross_validation parameters
        n_splits, n_repeats = cross_validation_config['n_splits'], cross_validation_config['n_repeats']
        random_state = cross_validation_config['random_state']
        
        # Initialize RepeatedKFold
        rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)

        print('--------------------------------')
        
        # Recover subjects id
        self.subject_ids = np.asarray(list(self.all_data.keys()))
        
        # Intra-subject learning
        if self.subject_ids.shape[0] < 3:
            subject_id = self.subject_ids[0]
            all_data = self.all_data[subject_id]
            all_labels = self.all_labels[subject_id]
            spike_events = self.all_spike_events[subject_id]
            index = [i for i in range(all_data.shape[0])]
            
            # Loop on the folds
            for fold, (train_ids, test_ids) in enumerate(rkf.split(index)):
            
                print(f'FOLD {fold}')
                print('--------------------------------')

                # Recover training and test datasets
                train_data, train_labels, train_spike_events = all_data[train_ids], all_labels[train_ids], spike_events[train_ids]
                test_data, test_labels, test_spike_events = all_data[test_ids], all_labels[test_ids], spike_events[test_ids]
                n_classes = len(np.unique(train_labels))
                
                # Apply CSP projection
                selected_rows = CSP_projection_config['selected_rows']
                CSP_projection = common_spatial_pattern(train_data, train_labels, selected_rows)
                #train_data = np.einsum('cd,ndt -> nct', CSP_projection, train_data)    
                #test_data = np.einsum('cd,ndt -> nct', CSP_projection, test_data) 

                # Z-score standardization 
                target_mean = np.mean(train_data)
                target_std = np.std(train_data)
                train_data = (train_data-target_mean) / target_std
                test_data = (test_data-target_mean) / target_std
                
                # Create dataloaders
                train_data = np.expand_dims(train_data, axis=1)
                test_data = np.expand_dims(test_data, axis=1)
                train_dataloader = get_dataloader(train_data, train_spike_events, batch_size, True, num_workers)
                test_dataloader = get_dataloader(test_data, test_spike_events, batch_size, False, num_workers)
                
                # Initialize neural network
                model = DetectionBertMEEG(**model_config)
                model.apply(reset_weights)
                
                # Move to gpu if available
                if available:
                    if torch.cuda.device_count() > 1:
                        model = torch.nn.DataParallel(model)
                model.to(device)
                
                # Define optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2),
                                             weight_decay=weight_decay, amsgrad=amsgrad)

                # Define warmup method
                if warmup:
                    warmup_scheduler = NoamOpt(lr, warmup, optimizer) 

                # Define scheduler
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor,patience=scheduler_patience,
                                                                          min_lr=min_lr, verbose=False)
                
                # Loop over the dataset n_epochs times
                for e in range(n_epochs):

                    # Train the model
                    train_loss = 0
                    train_steps = 0

                    # Set training mode
                    model.train()
                    for i, (data, labels) in enumerate(train_dataloader):

                        # Apply time window reduction
                        n_time_windows = model_config['n_time_windows']
                        labels = get_spike_windows(labels, n_time_windows)
                        if mix_up:

                            """
                            Apply a mix-up strategy for data augmentation inspired by:
                            `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
                            """

                            # Roll a copy of the batch
                            roll_factor =  torch.randint(0, data.shape[0], (1,)).item()
                            rolled_data = torch.roll(data, roll_factor, dims=0)        
                            rolled_labels = torch.roll(labels, roll_factor, dims=0)  

                            # Create a tensor of lambdas sampled from the beta distribution
                            lambdas = np.random.beta(BETA, BETA, data.shape[0])

                            # Trick inspired by `<https://forums.fast.ai/t/mixup-data-augmentation/22764>`
                            lambdas = torch.reshape(torch.tensor(np.maximum(lambdas, 1-lambdas)), (-1,1,1,1))

                            # Mix samples
                            mix_data = lambdas*data + (1-lambdas)*rolled_data

                            # Format conversion and move to device
                            mix_data = Variable(mix_data.type(self.Tensor), requires_grad=True)
                            data = Variable(data.type(self.Tensor), requires_grad=True)
                            labels = Variable(labels.type(self.LongTensor))
                            rolled_labels = Variable(rolled_labels.type(self.LongTensor))
                            mix_data, labels, rolled_labels = mix_data.to(device), labels.to(device), rolled_labels.to(device)

                            # Zero the parameter gradients
                            optimizer.zero_grad()

                            # Forward
                            _, mix_outputs = model(mix_data)
                            y_pred = torch.max(mix_outputs.data, -1)[1] 

                            # Concatenate the batches
                            stack_mix_outputs = rearrange(mix_outputs, '(b) (v) o -> (b v) o') 
                            stack_labels = rearrange(labels, '(b) (v) -> (b v)') 
                            stack_rolled_labels = rearrange(rolled_labels, '(b) (v) -> (b v)') 

                            # Compute mix-up loss
                            lambdas = lambdas.squeeze().to(device)
                            loss = (lambdas * self.train_criterion_cls(stack_mix_outputs, stack_labels) 
                                    + ((1-lambdas) * self.train_criterion_cls(stack_mix_outputs, stack_rolled_labels)))
                            loss = loss.sum()

                            # Backward
                            loss.backward()

                            # Update learning rate
                            if warmup:
                                warmup_scheduler.step()
                            else:
                                optimizer.step()

                            # Detach from device
                            loss = loss.cpu().detach().numpy()

                            # Recover training loss and confusion matrix
                            train_loss += loss
                            train_steps += 1

                        else:

                            # Format conversion and move to device
                            data = Variable(data.type(self.Tensor), requires_grad=True)
                            labels = Variable(labels.type(self.LongTensor))
                            data, labels = data.to(device), labels.to(device)

                            # Zero the parameter gradients
                            optimizer.zero_grad()

                            # Forward
                            _, outputs = model(data)

                            # Concatenate the batches
                            stack_outputs = rearrange(outputs, '(b) (v) o -> (b v) o') 
                            stack_labels = rearrange(labels, '(b) (v) -> (b v)') 

                            # Compute training loss
                            loss = self.train_criterion_cls(stack_outputs, stack_labels)

                            # Backward
                            loss.backward()

                            # Update learning rate
                            if warmup:
                                warmup_scheduler.step()
                            else:
                                optimizer.step()

                            # Detach from device
                            loss = loss.cpu().detach().numpy()

                            # Recover training loss and confusion matrix
                            train_loss += loss
                            train_steps += 1


                    # Print loss
                    train_loss /= train_steps
                    if (e+1)%5 == 0:
                        logger.info('Loss at epoch {}: {}'.format(e+1, train_loss))
   
                    # Update learning rate if training loss does not improve
                    if scheduler:
                        lr_scheduler.step(train_loss)
            
                # Training finished
                logger.info('Training process has finished. Saving trained model.')
                
                # Saving the model
                save_path = f'./model-fold-{fold}.pth'
                #torch.save(model.state_dict(), save_path)

                # Print about testing
                logger.info('Starting testing')

                # Set evaluation mode
                model.eval()

                # Initialize confusion_matrix
                confusion_matrix = np.zeros((2,2)) # Here the positive class is the class 1

                with torch.no_grad():
                    for i, (data, labels) in enumerate(test_dataloader):

                        # Apply time window reduction
                        labels = get_spike_windows(labels, n_time_windows)

                        # Format conversion and move to device
                        data = Variable(data.type(self.Tensor), requires_grad=True)
                        labels = Variable(labels.type(self.LongTensor))
                        data, labels = data.to(device), labels.to(device)

                        # Inference
                        _, outputs = model(data)
                        pred = torch.max(outputs.data, -1)[1]

                        # Detach from device
                        labels = labels.cpu().detach()
                        pred = pred.cpu().detach()

                        # Recover confusion matrix
                        for t, p in zip(labels.reshape(-1), pred.reshape(-1)):
                            confusion_matrix[t.long(), p.long()] += 1
                          
                    logger.info('Evaluation finished')
                    
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
                    F1_score = 2 * (Sensitivity*Precision) / (Sensitivity+Precision)
                    
                    # Print accuracy
                    print('Accuracy for fold {}: {}'.format(fold, Accuracy))
                    print('Specificity for fold {}: {}'.format(fold, Specificity))
                    print('Sensitivity for fold {}: {}'.format(fold, Sensitivity))
                    print('Precision for fold {}: {}'.format(fold, Precision))
                    print('F1_score for fold {}: {}'.format(fold, F1_score))
                    print('--------------------------------')
                    
                    # Recover metrics for fold
                    results[fold] = {'Accuracy':Accuracy, 'Specificity':Specificity, 'Sensitivity': Sensitivity, 'Precision':Precision, 'F1_score':F1_score}
                
            # Print fold results
            print(f'K-FOLD CROSS VALIDATION RESULTS FOR {n_splits*n_repeats} FOLDS')
            print('--------------------------------')
            avg_acc = np.mean([results[fold]['Accuracy'] for fold in range(n_splits*n_repeats)])
            avg_spec = np.mean([results[fold]['Specificity'] for fold in range(n_splits*n_repeats)])
            avg_sens = np.mean([results[fold]['Sensitivity'] for fold in range(n_splits*n_repeats)])
            avg_prec = np.mean([results[fold]['Precision'] for fold in range(n_splits*n_repeats)])
            avg_F1 = np.mean([results[fold]['F1_score'] for fold in range(n_splits*n_repeats)])
            results['Mean'] = {'Accuracy':avg_acc, 'Specificity':avg_spec, 'Sensitivity': avg_sens, 'Precision':avg_prec, 'F1_score':avg_F1}
            print('Average results: {}'.format(results['Mean']))

            return results
    
        # Cross-subject learning
        else:
            index = range(self.subject_ids.shape[0])
            for fold, (train_ids, test_ids) in enumerate(rkf.split(index)):
                
                # Print
                print(f'FOLD {fold}')
                print('--------------------------------')
                
                # recover training and test datasets
                train_subject_ids = self.subject_ids[train_ids]
                test_subject_ids = self.subject_ids[test_ids]
                train_data = np.concatenate([self.all_data[id] for id in train_subject_ids], axis=0)
                train_data = np.expand_dims(train_data, axis=1)
                train_labels = np.concatenate([self.all_labels[id] for id in train_subject_ids], axis=0)
                train_spike_events = np.concatenate([self.all_spike_events[id] for id in train_subject_ids], axis=0)
                train_dataloader = get_cross_subject_dataloader(train_data, train_spike_events, batch_size, True, num_workers)
                test_data = np.concatenate([self.all_data[id] for id in train_subject_ids], axis=0)
                test_data = np.expand_dims(test_data, axis=1)
                test_labels = np.concatenate([self.all_labels[id] for id in test_subject_ids], axis=0)
                test_spike_events = np.concatenate([self.all_spike_events[id] for id in test_subject_ids], axis=0)
                test_dataloader = get_cross_subject_dataloader(test_data, test_spike_events, batch_size, True, num_workers)

                # Initialize neural network
                model = DetectionBertMEEG(**model_config)
                model.apply(reset_weights)
                
                # Move to gpu if available
                if available:
                    if torch.cuda.device_count() > 1:
                        model = torch.nn.DataParallel(model)
                model.to(device)
                
                # Define optimizer
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2),
                                             weight_decay=weight_decay, amsgrad=amsgrad)

                # Define warmup method
                if warmup:
                    warmup_scheduler = NoamOpt(lr, warmup, optimizer) 

                # Define scheduler
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor,patience=scheduler_patience,
                                                                          min_lr=min_lr, verbose=False)
                
                # Loop over the dataset n_epochs times
                for e in range(n_epochs):

                    # Train the model
                    train_loss = 0
                    train_steps = 0

                    # Set training mode
                    model.train()
                    for i, (data, labels) in enumerate(train_dataloader):

                        # Apply time window reduction
                        n_time_windows = model_config['n_time_windows']
                        labels = get_spike_windows(labels, n_time_windows)
                        if mix_up:

                            """
                            Apply a mix-up strategy for data augmentation inspired by:
                            `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.
                            """

                            # Roll a copy of the batch
                            roll_factor =  torch.randint(0, data.shape[0], (1,)).item()
                            rolled_data = torch.roll(data, roll_factor, dims=0)        
                            rolled_labels = torch.roll(labels, roll_factor, dims=0)  

                            # Create a tensor of lambdas sampled from the beta distribution
                            lambdas = np.random.beta(BETA, BETA, data.shape[0])

                            # Trick inspired by `<https://forums.fast.ai/t/mixup-data-augmentation/22764>`
                            lambdas = torch.reshape(torch.tensor(np.maximum(lambdas, 1-lambdas)), (-1,1,1,1))

                            # Mix samples
                            mix_data = lambdas*data + (1-lambdas)*rolled_data

                            # Format conversion and move to device
                            mix_data = Variable(mix_data.type(self.Tensor), requires_grad=True)
                            data = Variable(data.type(self.Tensor), requires_grad=True)
                            labels = Variable(labels.type(self.LongTensor))
                            rolled_labels = Variable(rolled_labels.type(self.LongTensor))
                            mix_data, labels, rolled_labels = mix_data.to(device), labels.to(device), rolled_labels.to(device)

                            # Zero the parameter gradients
                            optimizer.zero_grad()

                            # Forward
                            _, mix_outputs = model(mix_data)
                            y_pred = torch.max(mix_outputs.data, -1)[1] 

                            # Concatenate the batches
                            stack_mix_outputs = rearrange(mix_outputs, '(b) (v) o -> (b v) o') 
                            stack_labels = rearrange(labels, '(b) (v) -> (b v)') 
                            stack_rolled_labels = rearrange(rolled_labels, '(b) (v) -> (b v)') 

                            # Compute mix-up loss
                            lambdas = lambdas.squeeze().to(device)
                            loss = (lambdas * self.train_criterion_cls(stack_mix_outputs, stack_labels) 
                                    + ((1-lambdas) * self.train_criterion_cls(stack_mix_outputs, stack_rolled_labels)))
                            loss = loss.sum()

                            # Backward
                            loss.backward()

                            # Update learning rate
                            if warmup:
                                warmup_scheduler.step()
                            else:
                                optimizer.step()

                            # Detach from device
                            loss = loss.cpu().detach().numpy()

                            # Recover training loss and confusion matrix
                            train_loss += loss
                            train_steps += 1

                        else:

                            # Format conversion and move to device
                            data = Variable(data.type(self.Tensor), requires_grad=True)
                            labels = Variable(labels.type(self.LongTensor))
                            data, labels = data.to(device), labels.to(device)

                            # Zero the parameter gradients
                            optimizer.zero_grad()

                            # Forward
                            _, outputs = model(data)

                            # Concatenate the batches
                            stack_outputs = rearrange(outputs, '(b) (v) o -> (b v) o') 
                            stack_labels = rearrange(labels, '(b) (v) -> (b v)') 

                            # Compute training loss
                            loss = self.train_criterion_cls(stack_outputs, stack_labels)

                            # Backward
                            loss.backward()

                            # Update learning rate
                            if warmup:
                                warmup_scheduler.step()
                            else:
                                optimizer.step()

                            # Detach from device
                            loss = loss.cpu().detach().numpy()

                            # Recover training loss and confusion matrix
                            train_loss += loss
                            train_steps += 1


                    # Print loss
                    train_loss /= train_steps
                    if (e+1)%5 == 0:
                        logger.info('Loss at epoch {}: {}'.format(e+1, train_loss))
   
                    # Update learning rate if training loss does not improve
                    if scheduler:
                        lr_scheduler.step(train_loss)
            
                # Training finished
                logger.info('Training process has finished. Saving trained model.')
                
                # Saving the model
                save_path = f'./model-fold-{fold}.pth'
                torch.save(model.state_dict(), save_path)

                # Print about testing
                logger.info('Starting testing')

                # Set evaluation mode
                model.eval()

                # Initialize confusion_matrix
                confusion_matrix = np.zeros((2,2)) # Here the positive class is the class 1

                with torch.no_grad():
                    for i, (data, labels) in enumerate(test_dataloader):

                        # Apply time window reduction
                        labels = get_spike_windows(labels, n_time_windows)

                        # Format conversion and move to device
                        data = Variable(data.type(self.Tensor), requires_grad=True)
                        labels = Variable(labels.type(self.LongTensor))
                        data, labels = data.to(device), labels.to(device)

                        # Inference
                        _, outputs = model(data)
                        pred = torch.max(outputs.data, -1)[1]

                        # Detach from device
                        labels = labels.cpu().detach()
                        pred = pred.cpu().detach()

                        # Recover confusion matrix
                        for t, p in zip(labels.reshape(-1), pred.reshape(-1)):
                            confusion_matrix[t.long(), p.long()] += 1
                          
                    logger.info('Evaluation finished')
                    
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
                    F1_score = 2 * (Sensitivity*Precision) / (Sensitivity+Precision)
                    
                    # Print accuracy
                    print('Accuracy for fold {}: {}'.format(fold, Accuracy))
                    print('Specificity for fold {}: {}'.format(fold, Specificity))
                    print('Sensitivity for fold {}: {}'.format(fold, Sensitivity))
                    print('Precision for fold {}: {}'.format(fold, Precision))
                    print('F1_score for fold {}: {}'.format(fold, F1_score))
                    print('--------------------------------')
                    
                    # Recover metrics for fold
                    results[fold] = {'Accuracy':Accuracy, 'Specificity':Specificity, 'Sensitivity': Sensitivity, 'Precision':Precision, 'F1_score':F1_score}
                
            # Print fold results
            print(f'K-FOLD CROSS VALIDATION RESULTS FOR {n_splits*n_repeats} FOLDS')
            print('--------------------------------')
            avg_acc = np.mean([results[fold]['Accuracy'] for fold in range(n_splits*n_repeats)])
            avg_spec = np.mean([results[fold]['Specificity'] for fold in range(n_splits*n_repeats)])
            avg_sens = np.mean([results[fold]['Sensitivity'] for fold in range(n_splits*n_repeats)])
            avg_prec = np.mean([results[fold]['Precision'] for fold in range(n_splits*n_repeats)])
            avg_F1 = np.mean([results[fold]['F1_score'] for fold in range(n_splits*n_repeats)])
            results['Mean'] = {'Accuracy':avg_acc, 'Specificity':avg_spec, 'Sensitivity': avg_sens, 'Precision':avg_prec, 'F1_score':avg_F1}
            print('Average results: {}'.format(results['Mean']))

            return results


    
    def evaluate(self, path_config, gpu_id):

        """
        Evaluate model on test set.

        Args:
            path_config (str): Path to the configuration file.
            gpu_id (int): Id of the cuda device to use if available.
            
        Return:
            test_info (dict): Contains metrics on test set.

        Example of config file:
        
            config = {
                'loader_parameters': {
                    'path_data': '../../Neuropoly_Internship/MEEG_data/EEG_signals/seconds_2_trials/',
                    'path_channel': '../../Neuropoly_Internship/MEEG_data/EEG_signals/channel_ctf_acc1.mat',
                    'wanted_event_label': 'saw_EST',
                    'wanted_channel_type': ['EEG'] ,
                    'binary_classification': True,
                    'selected_rows': 10,
                    'train_size': 0.6,
                    'test_size': 0.25,
                    'shuffle': True,
                    'random_state': 42
                    },
                'training_parameters': {
                    'batch_size': 8,
                    'num_workers': 0,
                    'Epochs': 100
                    'Mix-up': True,
                    'BETA': 0.4,
                    'use_cost_sensitive': True,
                    'lambda': 10,
                    'early_stopping': {
                        'patience': 20,
                        'checkpoint': 'detection_EEG_checkpoint.pt'
                        }
                    },
                'model': {
                    'n_channels': 20,
                    'n_time_points': 201,
                    'attention_dropout': 0.8,
                    'attention_kernel': 30,
                    'attention_stride': 5,
                    'spatial_dropout': 0.8,
                    'position_kernel': 20,
                    'position_stride': 1,
                    'emb_size': 30,
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
                'path_output': {
                    'folder_output': '../Trained_MEEG_models/spike_detection/',
                    'timeID': 2022_04_20-14_24_22,
                    'path_CSP_projection': '../Trained_MEEG_models/spike_detection/CSP_projection_2022_04_20-14_24_22.npy',
                    'path_config': '../Trained_MEEG_models/spike_detection/config_2022_04_20-14_24_22.json',
                    'path_model': '../Trained_MEEG_models/spike_detection/model_2022_04_20-14_24_22',
                    'path_training_info': '../Trained_MEEG_models/spike_detection/train_info_2022_04_20-14_24_22.csv',
                    'path_validation_info': '../Trained_MEEG_models/spike_detection/val_info_2022_04_20-14_24_22.csv',
                    'path_test_info': '../Trained_MEEG_models/spike_detection/test_info_2022_04_20-14_24_22.csv'
                    }
                } 
        """
              
        # Recover config files
        with open(path_config) as f:
            config = json.loads(f.read())
            
        # Recover paths
        path_model = config['path_output']['path_model']
        path_test_info = config['path_output']['path_test_info']
        
        # Recover model, optimizer and training configuration
        training_config, model_config, optimizer_config = config['training_parameters'], config['model'], config['optimizer']
            
        # Recover training parameters
        batch_size, num_workers = training_config['batch_size'], training_config['num_workers']
        
        # Create dataloader
        self.test_dataloader = get_dataloader(self.test_data, self.test_spike_events, batch_size, False, num_workers)

        # Define model
        n_time_windows = model_config['n_time_windows']
        model = DetectionBertMEEG(**model_config)
        
        # Move to gpu if available
        available, device = define_device(gpu_id)
        if available:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        model.to(device)
        
        # Load model state
        model.load_state_dict(torch.load(path_model))
        
        # Set evaluation mode
        model.eval()
        
        # Recover test metrics
        test_info = {"Accuracy": 0, "Specificity":0,"Sensitivity": 0, "Precision": 0, "F1_score": 0}
        
        # Initialize confusion_matrix
        confusion_matrix = np.zeros((2,2)) # Here the positive class is the class 1
        
        with torch.no_grad():
            for i, (data, labels) in enumerate(self.test_dataloader):

                # Apply time window reduction
                labels = get_spike_windows(labels, n_time_windows)
                
                # Format conversion and move to device
                data = Variable(data.type(self.Tensor), requires_grad=True)
                labels = Variable(labels.type(self.LongTensor))
                data, labels = data.to(device), labels.to(device)
            
                # Inference
                _, outputs = model(data)
                pred = torch.max(outputs.data, -1)[1]

                # Detach from device
                labels = labels.cpu().detach()
                pred = pred.cpu().detach()

                # Recover confusion matrix
                for t, p in zip(labels.reshape(-1), pred.reshape(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    
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
            F1_score = 2 * (Sensitivity*Precision) / (Sensitivity+Precision)
        
        test_info["Accuracy"] = Accuracy
        test_info["Specificity"] = Specificity
        test_info["Sensitivity"] = Sensitivity
        test_info["Precision"] = Precision 
        test_info["F1_score"] = F1_score

        logger.info('Evaluation finished')
        
        print('Average Accuracy on test set:', "%.4f" % round(Accuracy, 4))
        print('Average Specificity on test set:', "%.4f" % round(Specificity, 4))
        print('Average Sensitivity on test set:', "%.4f" % round(Sensitivity, 4))
        print('Average Precision on test set:', "%.4f" % round(Precision, 4))
        print('Average F1 score on test set:', "%.4f" % round(F1_score, 4), '\n')
        
        logger.info("Saving test information")

        test_df = pd.DataFrame([test_info])
        test_df.to_csv(path_test_info, index=False)
            
        print('Test information here:', path_test_info, '\n')
        
        return test_info

    
################ Return the Id of the time_window where there are spikes so that we can visualize them in Brainstorm #################
def spike_inference(path_config, data, path_output, gpu_id):
              
    # Recover config files
    with open(path_config) as f:
        config = json.loads(f.read())

    # Recover paths
    path_model = config['path_output']['path_model']

    # Recover model, optimizer and training configuration
    model_config = config['model']

    # Define model
    n_time_windows = model_config['n_time_windows']
    model = DetectionBertMEEG(**model_config)

    # Move to gpu if available
    available, device = define_device(gpu_id)
    if available:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    model.to(device)

    # Load model state
    model.load_state_dict(torch.load(path_model))

    # Set evaluation mode
    model.eval()
    
    # Create artifical labels: initialize them without any spikes
    n_time_points = data.shape[-1]
    labels = 0
    dataloader = get_dataloader(data, batch_size, False, num_workers)
    # Initialize trials and inference array
    trials = {}
    inferences = {}
    spikes = {}
    with torch.no_grad():
        for i, data in enumerate(dataloader):

            # Format conversion and move to device
            data = Variable(data.type(self.Tensor), requires_grad=True)
            data = data.to(device)

            # Inference
            _, outputs = model(data)
            pred = torch.max(outputs.data, -1)[1]
            
            # Detach from device
            pred = pred.cpu().detach()
            spike_index = np.where(pred.numpy() == 1)[0]
            
            # Recover inference
            trials[i] = data
            inferences[i] = pred
            spikes[i] = spike_index

    logger.info('Inference finished')

    return trials, inferences, spikes



def main(): 
    
    """
    Train or evaluate DetectionBertMEEG.
    
    Training: run the following command in terminal to train and save the trained model.

            python spike_detection.py --train --save --path-data [path to data] --path-channel [path to channel file] 
            --path-config [path to config file] --path-output [path to output folder] --gpu_id 0
                
        Example of config file: path_data and path_channel can be already written in the config file but do not need to.
        
            config = {
                'loader_parameters': {
                    'path_data': '../../Neuropoly_Internship/MEEG_data/EEG_signals/seconds_2_trials/',
                    'path_channel': '../../Neuropoly_Internship/MEEG_data/EEG_signals/channel_ctf_acc1.mat',
                    'wanted_event_label': 'saw_EST',
                    'wanted_channel_type': ['EEG'] ,
                    'binary_classification': True,
                    'selected_rows': 10,
                    'train_size': 0.6,
                    'test_size': 0.25,
                    'shuffle': True,
                    'random_state': 42
                    },
                'training_parameters': {
                    'batch_size': 8,
                    'num_workers': 0,
                    'Epochs': 100
                    'Mix-up': True,
                    'BETA': 0.4,
                    'use_cost_sensitive': True,
                    'lambda': 10,
                    'early_stopping': {
                        'patience': 20,
                        'checkpoint': 'detection_EEG_checkpoint.pt'
                        }
                    },
                'model': {
                    'n_channels': 20,
                    'n_time_points': 201,
                    'attention_dropout': 0.8,
                    'attention_kernel': 30,
                    'attention_stride': 5,
                    'spatial_dropout': 0.8,
                    'position_kernel': 20,
                    'position_stride': 1,
                    'emb_size': 30,
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
                }  
                
    Test: run the following command in terminal to test the trained model.

            python spike_detection.py --test --path-config [path to config file] --gpu_id 0
            
        Example of config file: path_data and path_channel must be already written in the config file.
        
            config = {
                'loader_parameters': {
                    'path_data': '../../Neuropoly_Internship/MEEG_data/EEG_signals/seconds_2_trials/',
                    'path_channel': '../../Neuropoly_Internship/MEEG_data/EEG_signals/channel_ctf_acc1.mat',
                    'wanted_event_label': 'saw_EST',
                    'wanted_channel_type': ['EEG'] ,
                    'binary_classification': True,
                    'selected_rows': 10,
                    'train_size': 0.6,
                    'test_size': 0.25,
                    'shuffle': True,
                    'random_state': 42
                    },
                'training_parameters': {
                    'batch_size': 8,
                    'num_workers': 0,
                    'Epochs': 100
                    'Mix-up': True,
                    'BETA': 0.4,
                    'use_cost_sensitive': True,
                    'lambda': 10,
                    'early_stopping': {
                        'patience': 20,
                        'checkpoint': 'detection_EEG_checkpoint.pt'
                        }
                    },
                'model': {
                    'n_channels': 20,
                    'n_time_points': 201,
                    'attention_dropout': 0.8,
                    'attention_kernel': 30,
                    'attention_stride': 5,
                    'spatial_dropout': 0.8,
                    'position_kernel': 20,
                    'position_stride': 1,
                    'emb_size': 30,
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
                'path_output': {
                    'folder_output': '../Trained_MEEG_models/spike_detection/',
                    'timeID': 2022_04_20-14_24_22,
                    'path_config': '../Trained_MEEG_models/spike_detection/config_2022_04_20-14_24_22.json',
                    'path_model': '../Trained_MEEG_models/spike_detection/model_2022_04_20-14_24_22',
                    'path_training_info': '../Trained_MEEG_models/spike_detection/train_info_2022_04_20-14_24_22.csv',
                    'path_validation_info': '../Trained_MEEG_models/spike_detection/val_info_2022_04_20-14_24_22.csv',
                    'path_test_info': '../Trained_MEEG_models/spike_detection/test_info_2022_04_20-14_24_22.csv'
                    }
                } 
    """
          
    parser = get_parser()
    args = parser.parse_args()
    
    # Recover commands
    train = args.train
    test = args.test

    # Recover data
    path_config = args.path_config

    # Recover save and gpu
    save = args.save
    gpu_id = args.gpu_id
        
    # Train model
    if train:
        path_data = args.path_data
        path_channel = args.path_channel
        
        # Recover config dictionnary and update path to data
        with open(path_config) as f:
            config = json.loads(f.read())
        config['loader_parameters']['path_data'] = path_data
        config['loader_parameters']['path_channel'] = path_channel

        # Initialize class DetectionTransformer
        detection = DetectionTransformer(config)
        path_output = args.path_output
        train_results = detection.train(config, save, path_output, gpu_id)
                  
    # Evaluate model
    elif test:
        
        # Recover config dictionnary and update path to data
        with open(path_config) as f:
            config = json.loads(f.read())

        # Initialize class DetectionTransformer
        detection = DetectionTransformer(config)
        test_results = detection.evaluate(path_config, gpu_id)


        
if __name__ == "__main__":
    main()
