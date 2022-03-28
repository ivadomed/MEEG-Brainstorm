#!/opt/anaconda3/bin/python

"""
This script is used to train and test the model. 
The EarlyStopping class is inspired from `<https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py>`.

Usage: type "from Train import <class>" to use one of its classes.
       type "from Train import <function>" to use one of its functions.
       
Contributors: Ambroise Odonnat.
"""

import json
import torch

import numpy as np

from os import listdir
from os.path import isfile, join
from torchmetrics.functional import  f1_score, precision_recall, specificity, average_precision, cohen_kappa 
from torch.autograd import Variable

from data import Data
from dataloader import train_test_dataset, get_dataloader
from parser import get_parser
from Model import Transformer_classification
from utils import *

from loguru import logger


class EarlyStopping:
    
    """
    Stop the training if validation loss doesn't improve after a given patience.
    Inspired from `<https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py>`.
    """
    
    def __init__(self, patience = 10, verbose = False, delta = 0, path = 'checkpoint.pt'):
        
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'         
        """
        
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        
    def save_checkpoint(self, val_loss, model):
        
        """
        Save model's parameters when validation loss decreases.
        """
        
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
        
class Trans():
    
    def __init__(self, folder, channel_fname, wanted_event_label,list_channel_type, binary_classification, selected_rows,\
                 train_size, validation_size, random_state, shuffle):
        
        """    
        Args:
            folder (list): List of paths to trial files (matlab dictionnaries),
            channel_fname (str): Path to channel file (matlab dictionnary),
            wanted_event_label (str): Annotation of wanted event,
            list_channel_type (list): List of the types of channels we want ,
            binary_classification (bool): Labellize trials in two classes as seizure/no seizure 
                                          instead of taking the number of seizures as label,
            selected_rows (int): Number of rows of each sub-spatial filter selected to create spatial filter,
            train_size (int): Size of the train set before separation into train and validation set,
            validation_size (int): Size of the validation set,
            random_state (int): Seed to insure reproductibility during shuffle in train_test_split function,
            shuffle (bool): Shuffle the data during train_test_split.
                                    
                                          
        """
        
        # Data format
        self.Tensor = torch.FloatTensor
        self.LongTensor = torch.LongTensor
        
        # Recover dataset
        logger.info("Get dataset")
        self.dataset = Data(folder,channel_fname,wanted_event_label, list_channel_type,binary_classification, selected_rows)
        self.allData, self.allLabels, self.allSpikeTimePoints, self.allTimes = self.dataset.csp_data()
        
        # Recover dataloader
        logger.info("Get dataloader")
        
        # Split data and labels in train, validation and test sets
        data_train, labels_train, data_test, labels_test = train_test_dataset(self.allData, self.allLabels,\
                                                                   train_size, shuffle, random_state)
        
        new_train_size = 1 - validation_size/train_size
        data_train, labels_train, data_val, labels_val = train_test_dataset(data_train, labels_train,\
                                                                   new_train_size, shuffle, random_state)
        
        self.data_train = np.expand_dims(data_train, axis = 1)
        self.data_val = np.expand_dims(data_val, axis = 1)
        self.data_test = np.expand_dims(data_test, axis = 1)
        self.labels_train = labels_train
        self.labels_val = labels_val
        self.labels_test= labels_test
        

    def train(self, config, model_path, config_path, weight_decay, l1_weight, l2_weight, scheduler, amsgrad, gpu_id, save):

        """
        Train the model and recover Loss, Accuracy, F1 scores and Average-precision curves on the training and validation set.
        We use an early stopping strategy which stops the training process when the validation loss does not improve anymore.

        Args:
            config (dict): Dictionnary of dictionnary containing model hyperparamaters, optimizer hyperparameters and training configuration,
            model_path (str): Path to save the model parameters,
            config_path (str): Path to save the training configuration file,
            weight_decay (float): Weight decay for L2 regularization in Adam optimizer,
            l1_weight (float): Weight for L1 regularization,
            l2_weight (float): Weight for L2 regularization,
            scheduler (bool): Use a scheduler on the learning rate,
            amsgrad (bool): Use AMSGrad instead of ADAM as optimizer algorithm,
            gpu_id (int): Index of cuda device to use if available,
            save (bool): Save model and optimizer parameters as well as training configuration file.

        Returns:
            tuple: train_info (dict): Values of Loss, F1 score, Precision, Recall on training set,
                   test_info (dict): Values of Loss, F1 score, Precision, Recall on validation set.
                   
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
        # Recover number of classes in the dataset
        self.n_classes = len(np.unique(self.allLabels))
        
        # Recover model, optimizer and training configuration
        training_config , model_config, optimizer_config = config['Training'], config['Model'], config['Optimizer']
        
        # Create dataloader
        batch_size, num_workers, balanced = training_config['batch_size'], training_config['num_workers'], training_config['balanced']
        self.train_dataloader = get_dataloader(self.data_train, self.labels_train, batch_size, num_workers, balanced)
        self.validation_dataloader = get_dataloader(self.data_val, self.labels_val, batch_size, num_workers, balanced) # set balanced = False because val set must reflect test set
        
        # Recover number of epochs
        n_epochs = training_config['Epochs']
        final_epoch = n_epochs
        
        # Define model
        self.model = Transformer_classification(**model_config)
        
        # Move to gpu if available
        available, device = define_device(gpu_id)
        if available:
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(model)
        self.model.to(device)

        # Define loss
        if available:
            self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.criterion_cls = torch.nn.CrossEntropyLoss()
                
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = optimizer_config['lr'],\
                                          betas=(optimizer_config['b1'], optimizer_config['b2']), weight_decay = weight_decay, amsgrad = amsgrad)
        
        # Define scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor = 0.5, patience = 20, min_lr = 1e-5, verbose = False)
        
        # Define early stopping
        early_stopping = EarlyStopping()
    
        # To recover loss, accuracy and F1 score
        train_info = dict((e,{"Loss": 0, "Accuracy": 0, "F1_score": 0, "Precision": 0, "Recall": 0}) for e in range(n_epochs))
        val_info = dict((e,{"Loss": 0, "Accuracy": 0, "F1_score": 0, "Precision": 0, "Recall": 0}) for e in range(n_epochs))
        
        # Loop over the dataset n_epochs times
        mix_up = training_config["Mix-up"]
        for e in range(n_epochs):
            
            # Train the model
            train_loss = 0
            correct, total = 0,0
            F1_train = 0
            precision_train, recall_train = 0,0
            train_steps = 0
            
            # Set training mode
            self.model.train()
            for i, (data, labels) in enumerate(self.train_dataloader):
                                
                if mix_up:
                    
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
                    loss = (lambdas.squeeze()).to(device)*self.criterion_cls(mix_outputs, labels) + (1-lambdas.squeeze()).to(device)*self.criterion_cls(mix_outputs, rolled_labels)
                    loss = loss.sum()
                    
                    # Apply Elastic Net regularization
                    cpu_model = self.model.to("cpu")
                    parameters = []
                    for parameter in cpu_model.parameters():
                        parameters.append(parameter.view(-1))
                    l1 = l1_weight * l1_regularization(torch.cat(parameters))
                    l2 = l2_weight * l2_regularization(torch.cat(parameters))
                    loss += l1
                    loss += l2
                    
                    loss.backward()
                    
                    # Optimize
                    self.optimizer.step()
                    
                    # Recover accurate prediction and F1 scores
                    train_loss += loss.cpu().detach().numpy()
                    y_pred = torch.max(mix_outputs.data, 1)[1]
                    correct += (y_pred == labels).sum().item()
                    total += labels.size(0)
                    F1_train += f1_score(y_pred.cpu().detach(), labels.cpu().detach(), average = 'weighted', num_classes = self.n_classes)
                    precision, recall = precision_recall(mix_outputs.cpu().detach(), labels.cpu().detach(), average = 'weighted',num_classes = self.n_classes)
                    precision_train += precision
                    recall_train += recall
                    train_steps += 1

                else:
                    data = Variable(data.type(self.Tensor), requires_grad = True)
                    labels = Variable(labels.type(self.LongTensor))
                    data, labels = data.to(device), labels.to(device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward
                    _, outputs = self.model(data)
                    loss = self.criterion_cls(outputs, labels)
                    
                    # Apply Elastic Net regularization
                    cpu_model = self.model.to("cpu")
                    parameters = []
                    for parameter in cpu_model.parameters():
                        parameters.append(parameter.view(-1))
                    l1 = l1_weight * l1_regularization(torch.cat(parameters))
                    l2 = l2_weight * l2_regularization(torch.cat(parameters))
                    loss += l1
                    loss += l2
                    
                    # backward
                    loss.backward()

                    # Optimize
                    self.optimizer.step()

                    # Recover accurate prediction and F1 scores
                    train_loss += loss.cpu().detach().numpy()
                    y_pred = torch.max(outputs.data, 1)[1]
                    correct += (y_pred == labels).sum().item()
                    total += labels.size(0)
                    F1_train += f1_score(y_pred.cpu().detach(), labels.cpu().detach(), average = 'weighted', num_classes = self.n_classes)
                    precision, recall = precision_recall(outputs.cpu().detach(), labels.cpu().detach(), average = 'weighted',num_classes = self.n_classes)
                    precision_train += precision
                    recall_train += recall
                    train_steps += 1

            # Recover accuracy and F1 score 
            train_info[e]["Loss"] = train_loss / train_steps
            train_info[e]["Accuracy"] = correct / total
            train_info[e]["F1_score"] = F1_train / train_steps
            train_info[e]["Precision"] = precision_train / train_steps
            train_info[e]["Recall"] = recall_train / train_steps
                
            # Evaluate the model
            val_loss = 0
            val_correct, val_total = 0,0
            F1_val = 0
            precision_val, recall_val = 0,0
            val_steps = 0
            
            # Set evaluation mode
            self.model.eval()
            for j, (val_data, val_labels) in enumerate(self.validation_dataloader):

                # Recover data, labels
                val_data = Variable(val_data.type(self.Tensor), requires_grad = True)
                val_labels = Variable(val_labels.type(self.LongTensor))
                val_data, val_labels = val_data.to(device), val_labels.to(device)

                # Recover outputs
                _, val_outputs = self.model(val_data)
                loss = self.criterion_cls(val_outputs, val_labels)

                # Recover accurate prediction and F1 scores
                val_loss += loss.cpu().detach().numpy()
                val_y_pred = torch.max(val_outputs, 1)[1]
                val_correct += (val_y_pred == val_labels).sum().item()
                val_total += val_labels.size(0)
                F1_val += f1_score(val_y_pred.cpu().detach(), val_labels.cpu().detach(), average = 'weighted', num_classes = self.n_classes)
                precision, recall = precision_recall(val_outputs.cpu().detach(), val_labels.cpu().detach(), average = 'weighted',num_classes = self.n_classes)
                precision_val += precision
                recall_val += recall
                val_steps += 1
                                
            # Update learning rate via the scheduler
            if scheduler:
                self.scheduler.step(val_loss / val_steps)
            
            # Recover accuracy and F1 score
            val_info[e]["Loss"] = val_loss / val_steps
            val_info[e]["Accuracy"] = val_correct / val_total
            val_info[e]["F1_score"] = F1_val / val_steps
            val_info[e]["Precision"] = precision_val / val_steps
            val_info[e]["Recall"] = recall_val / val_steps
                
            # Update early stopping
            early_stopping(val_loss / val_steps, self.model)
            if early_stopping.early_stop:
                final_epoch = e+1
                print("Early stopping at epoch: ", final_epoch)
                break      
              
        logger.info("Training finished")
        
        print('Mean Accuracy on validation set:', "%.4f" % round(np.float(np.mean([val_info[e]["Accuracy"] for e in range(final_epoch)])), 4))
        print('Mean F1 score on validation set:', "%.4f" % round(np.float(np.mean([val_info[e]["F1_score"] for e in range(final_epoch)])), 4))
        print('Mean Precision on validation set:', "%.4f" % round(np.float(np.mean([val_info[e]["Accuracy"] for e in range(final_epoch)])), 4))
        print('Mean Recall on validation set:', "%.4f" % round(np.float(np.mean([val_info[e]["Recall"] for e in range(final_epoch)])), 4))
        print('Final Accuracy on validation set:', "%.4f" % round(np.float(val_info[final_epoch-1]["Accuracy"]), 4))        
        print('Final F1 score on validation set:', "%.4f" % round(np.float(val_info[final_epoch-1]["F1_score"]), 4))
        
        if save:
            config['Training']['Epochs'] = final_epoch
            logger.info("Saving config files")
            json.dump(config, open(config_path, 'w' ))
            
            logger.info("Saving parameters")
            torch.save(torch.load('checkpoint.pt'), model_path)
            
            print('Model state here:', model_path)
            print('Config file here:', config_path, '\n')
                
        return train_info, val_info, final_epoch
    
    
    def evaluate(self, config_path, model_path, gpu_id):

        """
        Evaluate a model on test set.

        Args:
            config_path (str): Path to recover training configuration dictionnary (saved in .json format),
            model_path (str): Path to recover model hyperparameters,
            gpu_id (int): Index of cuda device to use if available.

        Returns:
            tuple: accuracy (float): Average accuracy on the test set,
                   F1_score (float): Average F1 score on the test set,
                   Weighted_F1_score (float): Average weighted F1 score on the test set.
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
        self.test_dataloader = get_dataloader(self.data_test, self.labels_test, batch_size, num_workers, balanced = False)

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
        F1_test = 0
        CK_test = 0
        Avg_precision_test = 0
        Precision = 0
        Recall = 0
        Specificity = 0
        steps= 0
        CK_steps = 0
        
        with torch.no_grad():
            for data, labels in self.test_dataloader:
                data = Variable(data.type(self.Tensor), requires_grad = True)
                labels = Variable(labels.type(self.LongTensor))
                data, labels = data.to(device), labels.to(device)
                _, outputs = model(data)
                y_pred = torch.max(outputs.data, 1)[1]
                correct += (y_pred == labels).sum().item()
                total += labels.size(0)
                F1_test += f1_score(y_pred.cpu().detach(), labels.cpu().detach(), average = 'weighted', num_classes = self.n_classes)
                Avg_precision_test += average_precision(outputs.cpu().detach(), labels.cpu().detach(), average = 'weighted',num_classes = self.n_classes)
                precision, recall = precision_recall(outputs.cpu().detach(), labels.cpu().detach(), average = 'weighted',num_classes = self.n_classes)
                Precision += precision
                Recall += recall
                Specificity += specificity(y_pred, labels, average = 'weighted', num_classes = self.n_classes)
                steps += 1
            
                # Count Cohen-Kappa score for prediction non equal to labels
                if sum(y_pred.cpu().detach().numpy() != labels.cpu().detach().numpy()) > 0:
                    CK_test += cohen_kappa(y_pred, labels, num_classes = self.n_classes)
                    CK_steps += 1

        Accuracy = correct / total
        F1_score = F1_test / steps
        Average_precision = Avg_precision_test / steps
        Precision /= steps
        Recall /= steps
        Specificity /= steps
        CK = CK_test / CK_steps
        
        logger.info('Evaluation finished')
        
        print('Accuracy on test set:', "%.4f" % round(np.float(Accuracy), 4))
        print('F1 score on test set:', "%.4f" % round(np.float(F1_score), 4))
        print('Precision on test set:', "%.4f" % round(np.float(Precision), 4))
        print('Recall on test set:', "%.4f" % round(np.float(Recall), 4))
        print('Specificity on test set:', "%.4f" % round(np.float(Specificity), 4))
        print('Average-precision on test set:', "%.4f" % round(np.float(Average_precision), 4))
        print('Cohen-Kappa coefficient on test set:', "%.4f" % round(np.float(CK), 4))
        
        return Accuracy , F1_score, Average_precision, Precision, Recall, Specificity, CK
        

def main(): 
   
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
    trans = Trans(folder, channel_fname, wanted_event_label,list_channel_type, binary_classification, selected_rows,\
                 train_size, validation_size, random_state, shuffle)
        
    # Train model
    if Train_bool:
        
        # Recover training commands
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
                                    amsgrad = amsgrad, gpu_id = gpu_id, save = save)
                  
    # Evaluate model
    if Test_bool:
        test_results = trans.evaluate(config_path, model_path, gpu_id)
        
        

if __name__ == "__main__":
    main()