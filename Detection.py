#!/usr/bin/env python

"""
This script is used to train and test the detection model. It detects spikes in EEG/MEG trials.

Usage: type "from Detection import <class>" to use one of its classes.
       
Contributors: Ambroise Odonnat.
"""

import json
import torch

import numpy as np

from os import listdir
from os.path import isfile, join
from torchmetrics.functional import  f1_score, precision_recall, specificity, average_precision, cohen_kappa, matthews_corrcoef
from torch.autograd import Variable
from torch import nn 
from einops import rearrange
from loguru import logger

from data import Data
from dataloader import get_dataloader
from parser import get_parser
from models import DetectionBertMEEG
from custom_losses import DetectionLoss
from early_stopping import EarlyStopping
from learning_rate_warmup import NoamOpt
from utils import *
                
    
class DetectionTransformer():
    
    def __init__(self, folder, channel_fname, wanted_event_label,list_channel_type, binary_classification, selected_rows,
                 train_size, test_size, shuffle, random_state):
        
        """    
        Args:
            folder (list): List of paths to trial files (matlab dictionnaries),
            channel_fname (str): Path to channel file (matlab dictionnary),
            wanted_event_label (str): Annotation of wanted event,
            list_channel_type (list): List of the types of channels we want ,
            binary_classification (bool): Labellize trials in two classes as seizure/no seizure 
                                          instead of taking the number of seizures as label,
            selected_rows (int): Number of rows of each sub-spatial filter selected to create spatial filter,
            normalization (bool): Normalize data after CSP projection,
            train_size (int): Size of the train set,
            test_size (int): Size of the test set,
            shuffle (bool): Shuffle the data during train_test_split,
            random_state (int): Seed to ensure reproductibility during shuffle in train_test_split function.
                                                                                     
        """
        
        # Data format
        self.Tensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor
        
        # Recover training, validation and testing datasets
        logger.info("Get dataset")
        
        self.dataset = Data(folder,channel_fname, wanted_event_label, list_channel_type,binary_classification,\
                            selected_rows, train_size, test_size, shuffle, random_state)
        self.n_classes, self.train_data, self.train_labels, self.train_spike_events,\
        self.val_data, self.val_labels, self.val_spike_events, self.test_data, self.test_labels, self.test_spike_events = self.dataset.csp_datasets()
        
        self.train_data = np.expand_dims(self.train_data, axis = 1)
        self.val_data = np.expand_dims(self.val_data, axis = 1)
        self.test_data = np.expand_dims(self.test_data, axis = 1)
   

    def train(self, config, cost_sensitive, lambd, weight_decay, amsgrad, scheduler, warmup, patience, gpu_id, save, model_path, config_path, checkpoint):

        """
        Train a detection model on the training and validation set.
        We use an early stopping strategy which stops the training process when the validation loss does not improve anymore.

        Args:
            config (dict): File containing training_configuration, model and optimizer hyperparameters,
            cost_sensitive (bool): Use cost-sensitive loss,
            lambd (float): Modulate the influence of the cost-sensitive weight,
            weight_decay (float): Value of weight decay in optimizer,
            amsgrad (bool): Use AMSGrad variant of Adam,
            scheduler (bool): Use ReduceLROnPLateau scheduler,
            warmup (int): If warmup > 0, apply warm-up steps on the learning rate,
            patience (int): Stop training if validation loss does not improve for more than patience epochs,
            gpu_id (int): Index of cuda device to use if available,
            save (bool): Save configuration file and model parameters,
            model_path (str): Path to save the model parameters,
            config_path (str): Path to save the configuration file,
            checkpoint (str): Path to model parameters checkpoint during training.

        Returns:
            tuple: train_info (dict): Values of Loss, Accuracy, macro and weighted F1 score on training set,
                   test_info (dict): Values of Loss, Accuracy, macro and weighted F1 score on validation set.
                   
        Example of configuration file:
        
            config = 
            {
            "Model": 
                    {
                    "normalized_shape": 201,
                    "linear_size": 28,
                    "vector_size": 201,
                    "attention_dropout": 0.4,
                    "attention_negative_slope": 0.01,
                    "attention_kernel_size": 40,
                    "attention_stride": 1,
                    "spatial_dropout": 0.5,
                    "out_channels": 2,
                    "position_kernel_size": 101,
                    "position_stride": 1,
                    "emb_negative_slope": 0.001,
                    "channel_kernel_size": 28,
                    "time_kernel_size": 40,
                    "time_stride": 1,
                    "slice_size": 15,
                    "depth": 5,
                    "num_heads": 5,
                    "transformer_dropout": 0.7,
                    "forward_expansion": 4,
                    "forward_dropout": 0.6,
                    "n_classes": 7
                    },
            "Optimizer": 
                    {
                    "lr": 0.01,
                    "b1": 0.9,
                    "b2": 0.999
                    }, 
            "Training": 
                    {
                    "batch_size": 4,
                    "num_workers": 4,
                    "balanced": true,
                    "Epochs": 50,
                    "Mix-up": False,
                    "BETA": 0.6
                    }
            }
        """
        
        # Recover model, optimizer and training configuration
        training_config , model_config, optimizer_config = config['Training'], config['Model'], config['Optimizer']

        # Recover number of epochs
        n_epochs = training_config['Epochs']
        final_epoch = n_epochs
        mix_up = training_config["Mix-up"]
        
        # Create dataloader
        batch_size, num_workers, balanced = training_config['batch_size'], training_config['num_workers'], training_config['balanced']
        self.train_dataloader = get_dataloader(self.train_data, self.train_spike_events, batch_size, num_workers)
        self.validation_dataloader = get_dataloader(self.val_data, self.val_spike_events, batch_size, num_workers) 

        # Define model
        self.model = DetectionBertMEEG(**model_config)
        n_time_windows = model_config['n_time_windows']
        
        # Move to gpu if available
        available, device = define_device(gpu_id)
        if available:
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(model)
        self.model.to(device)

        # Define training and validation losses
        if cost_sensitive:
            self.train_criterion_cls = DetectionLoss(torch.nn.CrossEntropyLoss(), 2, lambd) 
        else:
            self.train_criterion_cls = torch.nn.CrossEntropyLoss()
        self.val_criterion_cls = torch.nn.CrossEntropyLoss()

        if available:
            self.train_criterion_cls = self.train_criterion_cls.cuda()
            self.val_criterion_cls = self.val_criterion_cls.cuda()

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = optimizer_config['lr'],betas = (optimizer_config['b1'], optimizer_config['b2']),\
                                          weight_decay = weight_decay, amsgrad = amsgrad)

        # Define warmup method
        if warmup:
            self.warmup_scheduler = NoamOpt(optimizer_config['lr'], warmup, self.optimizer) 

        # Define scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, patience = 20, min_lr = 1e-5, verbose = False)

        # Define early stopping
        self.checkpoint = checkpoint
        early_stopping = EarlyStopping(patience = patience, path = self.checkpoint)

        # Recover loss, accuracy and F1 score
        train_info = dict((e,{"Loss": 0, "F1_score": 0,"Sensitivity": 0, "Precision": 0}) for e in range(n_epochs))
        val_info = dict((e,{"Loss": 0, "F1_score": 0,"Sensitivity": 0, "Precision": 0}) for e in range(n_epochs))

        # Loop over the dataset n_epochs times
        for e in range(n_epochs):

            # Train the model
            train_loss = 0
            confusion_matrix = np.zeros((2,2))
            train_steps = 0

            # Set training mode
            self.model.train()
            for i, (data, labels) in enumerate(self.train_dataloader):

                if mix_up:

                    # Apply time window reduction
                    labels = get_spike_windows(labels, n_time_windows)
                    
                    # Apply a mix-up strategy for data augmentation as adviced here '<https://forums.fast.ai/t/mixup-data-augmentation/22764>'        
                    BETA = training_config["BETA"]

                    # Roll a copy of the batch
                    roll_factor =  torch.randint(0, data.shape[0], (1,)).item()
                    rolled_data = torch.roll(data, roll_factor, dims=0)        
                    rolled_labels = torch.roll(labels, roll_factor, dims=0)  

                    # Create a tensor of lambdas sampled from the beta distribution
                    lambdas = np.random.beta(BETA, BETA, data.shape[0])

                    # trick from here https://forums.fast.ai/t/mixup-data-augmentation/22764
                    lambdas = torch.reshape(torch.tensor(np.maximum(lambdas, 1-lambdas)), (-1,1,1,1))

                    # Mix samples
                    mix_data = lambdas*data + (1-lambdas)*rolled_data

                    # Recover data, labels
                    mix_data = Variable(mix_data.type(self.Tensor), requires_grad = True)
                    data = Variable(data.type(self.Tensor), requires_grad = True)
                    labels = Variable(labels.type(self.LongTensor))
                    rolled_labels = Variable(rolled_labels.type(self.LongTensor))
                    mix_data, labels, rolled_labels = mix_data.to(device), labels.to(device), rolled_labels.to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward
                    _, mix_outputs = self.model(mix_data)
                    stack_mix_outputs = rearrange(mix_outputs, '(b) (v) o -> (b v) o') 
                    stack_labels = rearrange(labels, '(b) (v) -> (b v)') 
                    stack_rolled_labels = rearrange(rolled_labels, '(b) (v) -> (b v)') 
                    loss = (lambdas.squeeze()).to(device)*self.train_criterion_cls(stack_mix_outputs, stack_labels) +\
                    (1-lambdas.squeeze()).to(device)*self.train_criterion_cls(stack_mix_outputs, stack_rolled_labels)
                    loss = loss.sum()
                    
                    # Backward
                    loss.backward()

                    # Update learning rate
                    if warmup:
                        self.warmup_scheduler.step()
                    else:
                        self.optimizer.step()

                    # Recover confusion matrix
                    train_loss += loss.cpu().detach().numpy()
                    y_pred = torch.max(mix_outputs.data, -1)[1]   
                    train_steps += 1
                    for t, p in zip(labels.reshape(-1), y_pred.reshape(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
                    
                else:
                    data = Variable(data.type(self.Tensor), requires_grad = True)
                    labels = get_spike_windows(labels, n_time_windows)
                    labels = Variable(labels.type(self.LongTensor))
                    data, labels = data.to(device), labels.to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward
                    _, outputs = self.model(data)
                    stack_outputs = rearrange(outputs, '(b) (v) o -> (b v) o') 
                    stack_labels = rearrange(labels, '(b) (v) -> (b v)') 
                    loss = self.train_criterion_cls(stack_outputs, stack_labels)

                    # backward
                    loss.backward()

                    # Update learning rate
                    if warmup:
                        self.warmup_scheduler.step()
                    else:
                        self.optimizer.step()

                    # Recover confusion matrix
                    train_loss += loss.cpu().detach().numpy()
                    y_pred = torch.max(outputs.data, -1)[1] 
                    train_steps += 1
                    for t, p in zip(labels.reshape(-1), y_pred.reshape(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
            
            TP = confusion_matrix[1][1] / train_steps
            TN = confusion_matrix[0][0] / train_steps
            FP = confusion_matrix[0][1] / train_steps
            FN = confusion_matrix[1][0] / train_steps
                
            P = TP + FN
            N = TN + FP
            
            Accuracy = (TP + TN) / (P + N)
            Specificity = round(TN / N,4)
            Sensitivity = round(TP / P,4)
            Precision = TP / (TP + FP)
            F1_score = 2 * (Sensitivity * Precision) / (Sensitivity + Precision)

            # Recover accuracy and F1 score 
            train_info[e]["Loss"] = train_loss / train_steps 
            train_info[e]["F1_score"] = F1_score
            train_info[e]["Sensitivity"] = Sensitivity
            train_info[e]["Precision"] = Precision

            # Evaluate the model
            val_loss = 0
            confusion_matrix = np.zeros((2,2))
            val_steps = 0

            # Set evaluation mode
            self.model.eval()
            for j, (val_data, val_labels) in enumerate(self.validation_dataloader):

                # Recover data, labels
                val_data = Variable(val_data.type(self.Tensor), requires_grad = True)
                val_labels = get_spike_windows(val_labels, n_time_windows)
                val_labels = Variable(val_labels.type(self.LongTensor))
                val_data, val_labels = val_data.to(device), val_labels.to(device)

                # Recover outputs
                _, val_outputs = self.model(val_data)
                stack_val_outputs = rearrange(val_outputs, '(b) (v) o -> (b v) o') 
                stack_val_labels = rearrange(val_labels, '(b) (v) -> (b v)') 
                loss = self.train_criterion_cls(stack_val_outputs, stack_val_labels)

                # Update learning rate if validation loss does not improve
                if scheduler:
                    self.lr_scheduler.step(loss.cpu().detach().numpy())

                # Recover confusion matrix
                val_loss += loss.cpu().detach().numpy()
                val_y_pred = torch.max(val_outputs.data, -1)[1]   
                val_steps += 1
                for t, p in zip(val_labels.reshape(-1), val_y_pred.reshape(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
                    
            TP = confusion_matrix[1][1]
            TN = confusion_matrix[0][0] 
            FP = confusion_matrix[0][1] 
            FN = confusion_matrix[1][0] 

            P = TP + FN
            N = TN + FP
            
            Accuracy = round((TP + TN) / (P + N),4)
            Specificity = round(TN / N,4)
            Sensitivity = round(TP / P,4)
            Precision = round(TP / (TP + FP),4)
            F1_score = 2 * (Sensitivity * Precision) / (Sensitivity + Precision)
            
            # Recover loss and metrics
            val_info[e]["Loss"] = val_loss / val_steps
            val_info[e]["F1_score"] = F1_score
            val_info[e]["Sensitivity"] = Sensitivity
            val_info[e]["Precision"] = Precision
            
            if e%10 == 0:
                print('validation epoch: ',e,'P: ',P, 'N: ',N,'TP: ',TP,'TN: ',TN,'FP: ',FP,'FN: ',FN)
                print('Accuracy: ',Accuracy, 'Specificity: ',Specificity, 'Sensitivity: ',Sensitivity, 'Precision: ',Precision, 'F1_score: ',F1_score)

            # Update early stopping
            early_stopping(val_loss / val_steps, self.model)
            if early_stopping.early_stop:
                final_epoch = e+1
                print("Early stopping at epoch: ", final_epoch)
                break      

        logger.info("Training finished")     
        
        if save:
            config['Training']['Epochs'] = final_epoch
            logger.info("Saving config files")
            json.dump(config, open(config_path, 'w' ))
            
            logger.info("Saving parameters")
            torch.save(torch.load(self.checkpoint), model_path)
            
            print('Model state here:', model_path)
            print('Config file here:', config_path, '\n')
                
        return train_info, val_info, final_epoch
    
    
    def evaluate(self, config_path, model_path, gpu_id):

        """
        Evaluate classification model on test set.

        Args:
            config_path (str): Path to recover training configuration dictionnary (saved in .json format),
            model_path (str): Path to recover model hyperparameters,
            gpu_id (int): Index of cuda device to use if available.

        Returns:
            tuple: Accuracy (float): Mean accuracy on the test set,
                   F1_score (float): Mean F1 score on the test set,
                   Precision (float): Mean Precision on the test set,
                   Recall (float): Mean Recall on the test set,
                   Specificity (float): Mean Specificity on the test set,
                   weighted_F1_score (float): Mean weighted F1 score on the test set,
                   weighted_Precision (float): Mean weighted Precision on the test set,
                   weighted_Recall (float): Mean weighted Recall on the test set,
                   weighted_Specificity (float): Mean weighted Specificity on the test set,
                   Average_precision (float): Mean Average-precision on the test set,
                   weighted_Average_precision (float): Mean weighted Average-precision on the test set,
                   MCC (float): Matthew's correlation coefficient on the test set.
        """
        
        # Recover number of classes in the dataset
        self.n_classes = len(np.unique(self.allLabels))
        
        # Recover config files
        with open(config_path) as f:
            config = json.loads(f.read())
            
        # Recover model, optimizer and training configuration
        training_config, model_config, optimizer_config = config['Training'], config['Model'], config['Optimizer']
            
        # Create dataloader
        batch_size, num_workers = training_config['batch_size'], training_config['num_workers']
        self.test_dataloader = get_dataloader(self.test_data, self.test_labels, batch_size, num_workers, balanced = False)

        # Load model
        model = Transformer_classification(**model_config)
        
        # Move to gpu if available
        available, device = define_device(gpu_id)
        if available:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        model.to(device)
        
        # Load model state
        model.load_state_dict(torch.load(model_path))
        
        # Set evaluation mode
        model.eval()
        
        # Initialize accuracy
        correct, total = 0,0
        F1_test, weighted_F1_test = 0,0
        Precision, weighted_Precision = 0,0
        Recall, weighted_Recall = 0,0
        Specificity, weighted_Specificity = 0,0
        Avg_precision_test = 0
        MCC_test = 0
        steps = 0
        
        with torch.no_grad():
            for data, labels in self.test_dataloader:
                data = Variable(data.type(self.Tensor), requires_grad = True)
                labels = Variable(labels.type(self.LongTensor))
                data, labels = data.to(device), labels.to(device)
                _, outputs = model(data)
                y_pred = torch.max(outputs.data, 1)[1]
                correct += (y_pred == labels).sum().item()
                total += labels.size(0)
                F1_test += f1_score(y_pred.cpu().detach(), labels.cpu().detach(), average = 'macro', num_classes = self.n_classes)
                weighted_F1_test += f1_score(y_pred.cpu().detach(), labels.cpu().detach(), average = 'weighted', num_classes = self.n_classes)
                precision, recall = precision_recall(outputs.cpu().detach(), labels.cpu().detach(), average = 'macro',num_classes = self.n_classes)
                Precision += precision
                Recall += recall
                weighted_precision, weighted_recall = precision_recall(outputs.cpu().detach(), labels.cpu().detach(), average = 'weighted',num_classes = self.n_classes)
                weighted_Precision += weighted_precision
                weighted_Recall += weighted_recall
                Specificity += specificity(y_pred, labels, average = 'macro', num_classes = self.n_classes)
                weighted_Specificity += specificity(y_pred, labels, average = 'weighted', num_classes = self.n_classes)
                Avg_precision_test += average_precision(outputs.cpu().detach(), labels.cpu().detach(), average = 'macro',num_classes = self.n_classes)
                MCC_test += matthews_corrcoef(y_pred, labels, num_classes = self.n_classes)
                steps += 1

        Accuracy = correct / total
        F1_score = F1_test / steps
        weighted_F1_score = weighted_F1_test / steps
        Precision /= steps
        weighted_Precision /= steps
        Recall /= steps
        weighted_Recall /= steps
        Specificity /= steps
        weighted_Specificity /= steps
        Average_precision = Avg_precision_test / steps
        MCC = MCC_test / steps
        
        logger.info('Evaluation finished')
        
        print('Accuracy on test set:', "%.4f" % round(np.float(Accuracy), 4))
        print('F1 score on test set:', "%.4f" % round(np.float(F1_score), 4))
        print('Precision on test set:', "%.4f" % round(np.float(Precision), 4))
        print('Recall on test set:', "%.4f" % round(np.float(Recall), 4))
        print('Specificity on test set:', "%.4f" % round(np.float(Specificity), 4))
        print('Weighted F1 score on test set:', "%.4f" % round(np.float(weighted_F1_score), 4))
        print('Weighted Precision on test set:', "%.4f" % round(np.float(weighted_Precision), 4))
        print('Weighted Recall on test set:', "%.4f" % round(np.float(weighted_Recall), 4))
        print('Weighted Specificity on test set:', "%.4f" % round(np.float(weighted_Specificity), 4))
        print('Average-precision on test set:', "%.4f" % round(np.float(Average_precision), 4))
        print('Matthew\'s correlation coefficient on test set:', "%.4f" % round(np.float(MCC), 4))
        
        return Accuracy , F1_score, Precision, Recall, Specificity, weighted_F1_score, weighted_Precision, weighted_Recall,\
               weighted_Specificity, Average_precision, MCC
        
        

def main(patience = 10): 
   
    """
    Train or evaluate model depending on args given in command line.
    All configuration files are dictionnaries saved in .json format.
    
    Training: To train and save the model, using a scheduler for the learning rate with optimizer AMSGrad, run the following command in your terminal:

            python Train.py --train --path-data [path to the data] --path-config_data [path to configuration file for data] 
            --path-config_training [path to configuration file for training] --path-model [path to save model parameters]
            --path-config [path to save training configuration file] -gpu [index of the cuda device to use if available]
            --save --scheduler --amsgrad -weight_decay [weight decay value] -l1_weight [L1 regularization weight value]
            -l2_weight [L2 regularization weight value]
                
    Testing: To evaluate the model, run the following command in your terminal:
            
            python Train.py --test --path-data [path to the data] --path-config_data 
            [path to configuration file for data] --path-config_training [path to configuration file for training] --path-model 
            [path to save model parameters] --path-config [path to save training configuration file] -gpu [index of the cuda device to use if available]
            
    Examples of configuration files:
    
            Data config file (dict):
        
            data_config = 
            {
            'channel_fname': '../../Neuropoly_Internship/MEEG_data/EEG_signals/channel_ctf_acc1.mat',
            'wanted_event_label': 'saw_EST',
            'channel_type': ['EEG'],
            'binary_classification': False,
            'selected_rows': 2,
            'train_size': 0.75,
            'validation_size': 0.15,
            'random_state': 42,
            'shuffle': True
            }
            
            Training config file (dict):
            
            config = 
            {
            "Model": 
                    {
                    "normalized_shape": 201,
                    "linear_size": 28,
                    "vector_size": 201,
                    "attention_dropout": 0.4,
                    "attention_negative_slope": 0.01,
                    "attention_kernel_size": 40,
                    "attention_stride": 1,
                    "spatial_dropout": 0.5,
                    "out_channels": 2,
                    "position_kernel_size": 101,
                    "position_stride": 1,
                    "emb_negative_slope": 0.001,
                    "channel_kernel_size": 28,
                    "time_kernel_size": 40,
                    "time_stride": 1,
                    "slice_size": 15,
                    "depth": 5,
                    "num_heads": 5,
                    "transformer_dropout": 0.7,
                    "forward_expansion": 4,
                    "forward_dropout": 0.6,
                    "n_classes": 7
                    },
            "Optimizer": 
                    {
                    "lr": 0.01,
                    "b1": 0.9,
                    "b2": 0.999
                    }, 
            "Training": 
                    {
                    "batch_size": 4,
                    "num_workers": 4,
                    "balanced": true,
                    "Epochs": 50,
                    "Mix-up": False,
                    "BETA": 0.6
                    }
            }
    """
    
    parser = get_parser()
    args = parser.parse_args()
    
    # Recover data and configuration files
    data_path = args.path_data
    data_config_path = args.path_config_data
    training_config_path = args.path_config_training
    model_path = args.path_model
    config_path = args.path_config

    # Recover commands
    Train_bool = args.train
    Test_bool = args.test

    # Recover gpu_id
    gpu_id = args.gpu_id
    
    # Recover data
    folder = [data_path+f for f in listdir(data_path) if isfile(join(data_path, f))]

    # Recover data config dictionnary
    with open(data_config_path) as f:
        data_config = json.loads(f.read())
        
    channel_fname = data_config['channel_fname']
    wanted_event_label = data_config['wanted_event_label']
    list_channel_type = data_config['channel_type']
    binary_classification = data_config['binary_classification']
    selected_rows = data_config['selected_rows']
    train_size = data_config['train_size']
    validation_size = data_config['validation_size']
    random_state = data_config['random_state']
    shuffle = data_config['shuffle']

    # Initialize class Trans
    trans = Transformer_classification(folder, channel_fname, wanted_event_label,list_channel_type, binary_classification, selected_rows,\
                 train_size, validation_size, random_state, shuffle)
        
    # Train model
    if Train_bool:
        
        # Recover training commands
        ############## ADD NEW ARGUMENTS IN MAIN FUNCTION !!!!!!!!! ###############
        save = args.save
        scheduler = args.scheduler
        amsgrad = args.amsgrad
    
        # Recover weight for L1 and L2 regularization
        weight_decay = args.weight_decay
        l1_weight = args.l1_weight
        l2_weight = args.l2_weight
    
        # Recover training config dictionnary
        with open(training_config_path) as f:
            config = json.loads(f.read())
            
        train_results = trans.train(config, model_path, config_path, weight_decay, l1_weight, l2_weight, scheduler = scheduler,\
                                    patience = patience, amsgrad = amsgrad, gpu_id = gpu_id, save = save)
                  
    # Evaluate model
    if Test_bool:
        test_results = trans.evaluate(config_path, model_path, gpu_id)
        
        

if __name__ == "__main__":
    main()