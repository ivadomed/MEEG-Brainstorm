#!/opt/anaconda3/bin/python

"""
This script is used to train and test the model. 

Usage: type "from Train import <class>" to use one of its classes.
       type "from Train import <function>" to use one of its functions.
       
Contributors: Ambroise Odonnat.
"""

import json
import torch

import numpy as np

from os import listdir
from os.path import isfile, join
from sklearn.metrics import f1_score
from torch.autograd import Variable

from data import Data
from dataloader import train_test_dataset, get_dataloader
from parser import get_parser
from Model import Transformer_classification
from utils import *

from loguru import logger



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
        

    def train(self, config, model_path, optimizer_path, config_path, weight_decay, l1_weight, l2_weight, gpu_id, save):

        """
        Train the model and plot loss, accuracy and F1 scores on the training and validation set.

        Args:
            config (dict): Dictionnary of dictionnary containing model hyperparamaters, optimizer hyperparameters and training configuration,
            model_path (str): Path to save the model parameters,
            optimizer_path (str): Path to save the opimizer parameters,
            config_path (str): Path to save the training configuration file,
            weight_decay (float): Weight decay for L2 regularization in Adam optimizer,
            l1_weight (float): Weight for L1 regularization,
            l2_weight (float): Weight for L2 regularization,
            gpu_id (int): Index of cuda device to use if available,
            save (bool): Save model and optimizer parameters as well as training configuration file.

        Returns:
            tuple: train_info (dict): Values of loss, accuracy, F1 score on training set,
                   test_info (dict): Values of loss, accuracy, f1 score on validation set.
                   
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
        
        # Create dataloader
        batch_size, num_workers, balanced = training_config['batch_size'], training_config['num_workers'], training_config['balanced']
        self.train_dataloader = get_dataloader(self.data_train, self.labels_train, batch_size, num_workers, balanced)
        self.validation_dataloader = get_dataloader(self.data_val, self.labels_val, batch_size, num_workers, balanced) # set balanced = False because val set must reflect test set
        
        # Recover number of epochs
        n_epochs = training_config['Epochs']
        
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
        # We apply a weight decay for L2 regularization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = optimizer_config['lr'],\
                                          betas=(optimizer_config['b1'], optimizer_config['b2']), weight_decay = weight_decay)
        
        # Recover loss, accuracy and F1 score
        train_info = dict((e,{"Loss": 0, "Accuracy": 0, "F1_score": 0}) for e in range(n_epochs))
        test_info = dict((e,{"Loss": 0, "Accuracy": 0, "F1_score": 0}) for e in range(n_epochs))
      
        # Loop over the dataset n_epochs times
        mix_up = training_config["Mix-up"]
        for e in range(n_epochs):
            
            # Train the model
            train_loss = 0
            correct, total = 0,0
            f1_train = 0
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
                    f1_train += f1_score(labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average = 'macro')
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
                    f1_train += f1_score(labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average = 'macro')
                    train_steps += 1

            # Recover accuracy and F1 score 
            train_info[e]["Loss"] = train_loss / train_steps
            train_info[e]["Accuracy"] = correct / total
            train_info[e]["F1_score"] = f1_train / train_steps
                
            # Evaluate the model
            test_loss = 0
            test_correct, test_total = 0,0
            f1_test = 0
            test_steps = 0
            
            # Set evaluation mode
            self.model.eval()
            
            for j, (test_data, test_labels) in enumerate(self.validation_dataloader):

                # Recover data, labels
                test_data = Variable(test_data.type(self.Tensor), requires_grad = True)
                test_labels = Variable(test_labels.type(self.LongTensor))
                test_data, test_labels = test_data.to(device), test_labels.to(device)

                # Recover outputs
                _, test_outputs = self.model(test_data)
                loss = self.criterion_cls(test_outputs, test_labels)

                # Recover accurate prediction and F1 scores
                test_loss += loss.cpu().detach().numpy()
                test_y_pred = torch.max(test_outputs, 1)[1]
                test_total += test_labels.size(0)
                test_correct += (test_y_pred == test_labels).sum().item()
                f1_test += f1_score(test_labels.cpu().detach().numpy(), test_y_pred.cpu().detach().numpy(), average = 'macro')
                test_steps += 1
                
            # Recover accuracy and F1 score
            test_info[e]["Loss"] = test_loss / test_steps
            test_info[e]["Accuracy"] = test_correct / test_total
            test_info[e]["F1_score"] = f1_test / test_steps
            
        logger.info("Training finished")
        print('The average validation accuracy is:', np.mean([test_info[e]["Accuracy"] for e in range(n_epochs)]))
        print('The average validation F1 score is:', np.mean([test_info[e]["F1_score"] for e in range(n_epochs)]))
        
        if save:
            logger.info("Saving config files")
            json.dump(config, open(config_path, 'w' ))
            
            logger.info("Saving parameters")
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            
            print('Model state here:', model_path)
            print('Optimizer state here:', optimizer_path)
            print('Config file here:', config_path, '\n')
                
        return train_info, test_info
    
    
    def evaluate(self, config_path, model_path, optimizer_path, gpu_id):

        """
        Evaluate a model on test set.

        Args:
            config_path (str): Path to recover training configuration dictionnary (saved in .json format),
            model_path (str): Path to recover model hyperparameters,
            optimizer_path (str): Path to recover optimizer parameters,
            gpu_id (int): Index of cuda device to use if available.

        Returns:
            tuple: accuracy (float): Average accuracy on the test set,
                   F1_score (float): Average F1 score on the test set.
        """
        
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
        
        # Load optimizer parameters
        optimizer = torch.optim.Adam(model.parameters(), lr = optimizer_config['lr'],\
                                          betas=(optimizer_config['b1'], optimizer_config['b2']))        
        optimizer.load_state_dict(torch.load(optimizer_path))
        
        # Initialize accuracy
        correct = 0
        total = 0
        f1_test = []
        with torch.no_grad():
            for data, labels in self.test_dataloader:
                data = Variable(data.type(self.Tensor), requires_grad = True)
                labels = Variable(labels.type(self.LongTensor))
                data, labels = data.to(device), labels.to(device)
                _, outputs = model(data)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                f1_test.append(f1_score(labels.cpu().detach().numpy(), predicted.cpu().detach().numpy(), average = 'macro'))

        accuracy = correct / total 
        F1_score = np.mean(f1_test)
        
        logger.info('Evaluation finished')
        print("Accuracy on test set: {} \nF1 score on test set: {}".format(accuracy, F1_score))
        
        return accuracy , F1_score
        

def main(): 
   
    """
    Train or evaluate model depending on args given in command line.
    All configuration files are dictionnaries saved in .json format.
    
    Training: --save, -gpu, -weight_decay, -l1_weight, l2_weight are optional.
            To train and save the model, run the following command in your terminal:

            python Train.py --train --save --path-data [path to the data] --path-config_data 
            [path to configuration file for data] --path-config_training [path to configuration file for training] --path-model 
            [path to save model parameters] --path-optimizer [path to save optimizer parameters] --path-config     
            [path to save training configuration file] -gpu [index of the cuda device to use if available] -weight_decay [weight decay value]
            -l1_weight [L1 regularization weight value] -l2_weight [L2 regularization weight value]
                
    Testing: -gpu is optional.
            To evaluate the model, run the following command in your terminal:
            
            python Train.py --test --path-data [path to the data] --path-config_data 
            [path to configuration file for data] --path-config_training [path to configuration file for training] --path-model 
            [path to save model parameters] --path-optimizer [path to save optimizer parameters] --path-config 
            [path to save training configuration file] -gpu [index of the cuda device to use if available]
            
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
    
    # Recover data path
    data_path = args.path_data
    data_config_path = args.path_config_data
    training_config_path = args.path_config_training
    model_path = args.path_model
    optimizer_path = args.path_optimizer
    config_path = args.path_config

    # Recover command
    Train_bool = args.train
    Test_bool = args.test
    save = args.save
    
    # Recover gpu_id
    gpu_id = args.gpu_id

    # Recover weight for L1 and L2 regularization
    weight_decay = args.weight_decay
    l1_weight = args.l1_weight
    l2_weight = args.l2_weight
    
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
        
        # Recover training config dictionnary
        with open(training_config_path) as f:
            config = json.loads(f.read())
            
        save = args.save
        train_results = trans.train(config, model_path, optimizer_path, config_path, weight_decay, l1_weight, l2_weight, gpu_id = gpu_id, save = save)
        train_info, test_info = train_results
                  
    # Evaluate model
    if Test_bool:
        accuracy, F1_score = trans.evaluate(config_path, model_path, optimizer_path, gpu_id)
        
        

if __name__ == "__main__":
    main()