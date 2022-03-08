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
from Model import Transformer
from utils import check_balance, define_device

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
        

    def train(self, config, model_path, optimizer_path, config_path, gpu_id, save):

        """
        Train the model and compute accuracy and F1 scores on the validation set.

        Args:
            config (dict): Dictionnary of dictionnary containing model hyperparamaters, optimizer parameters 
                           and training configuration --> dataloaders parameters (batch size, number of workers),
                           number of epochs, mix-up parameters,
            model_path (str): Path to save the model parameters,
            optimizer_path (str): Path to save the opimizer parameters,
            config_path (str): Path to save the config,
            save (bool): Save information into the previous paths.

        Returns:
            tuple: train_info (dict): Values of loss, accuracy, F1 score on training set,
                   test_info (dict): Values of loss, accuracy, f1 score on validation set,
                   bestAcc (float): Best accuracy on validation set,
                   averAcc (float) Average accuracy on validation set,
                   bestF1 (float): Best F1 score on validation set,
                   averF1 (float) Average F1 score on validation set,
                   Y_true_acc (array): labels corresponding to best accuracy
                   Y_pred_acc (array): prediciton corresponding to best accuracy,
                   Y_true_F1 (array): labels corresponding to best F1 score,
                   Y_pred_F1 (array): prediciton corresponding to best F1 score.
        """
        
        # Recover model, optimizer and training configuration
        training_config , model_config, optimizer_config = config['Training'], config['Model'], config['Optimizer']
        
        # Create dataloader
        batch_size, num_workers, balanced = training_config['batch_size'], training_config['num_workers'], training_config['balanced']
        self.train_dataloader = get_dataloader(self.data_train, self.labels_train, batch_size, num_workers, balanced)
        self.validation_dataloader = get_dataloader(self.data_val, self.labels_val, batch_size, num_workers, balanced)
        
        # Recover number of epochs
        n_epochs = training_config['Epochs']
        
        # Define model
        self.model = Transformer(**model_config)
        
        # Move to gpu if available
        available, device = define_device(gpu_id)
        if available:
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
        self.model.to(device)

        # Define loss
        if available:
            self.criterion_cls = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.criterion_cls = torch.nn.CrossEntropyLoss()
            
                
        # Define optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = optimizer_config['lr'],\
                                          betas=(optimizer_config['b1'], optimizer_config['b2']))
        
        bestAcc = 0
        averAcc = 0
        bestF1 = 0
        averF1 = 0
        train_info = dict((e,{"Loss": 0, "Accuracy": 0, "F1_score": 0}) for e in range(n_epochs))
        test_info = dict((e,{"Loss": 0, "Accuracy": 0, "F1_score": 0}) for e in range(n_epochs))
        best_epochs_acc = 0
        best_epochs_F1 = 0
        num = 0
        Y_true_acc = []
        Y_pred_acc = [] 
        Y_true_F1 = []
        Y_pred_F1 = []   
      
        # Loop over the dataset n_epochs times
        mix_up = training_config["Mix-up"]
        for e in range(n_epochs):
            
            # Train the model
            correct, total = 0,0
            weighted_f1_train, f1_train = [],[]

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
                    mix_data = Variable(mix_data.type(self.Tensor))
                    data = Variable(data.type(self.Tensor))
                    labels = Variable(labels.type(self.LongTensor))
                    rolled_labels = Variable(rolled_labels.type(self.LongTensor))
                    mix_data, labels, rolled_labels = mix_data.to(device), labels.to(device), rolled_labels.to(device)
                    
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward
                    _, mix_outputs = self.model(mix_data)
                    loss = (lambdas.squeeze()).to(device)*self.criterion_cls(mix_outputs, labels) + (1-lambdas.squeeze()).to(device)*self.criterion_cls(mix_outputs, rolled_labels)
                    loss = loss.sum()
                    loss.backward()
                    
                    # Optimize
                    self.optimizer.step()
                    
                    # Recover accurate prediction and F1 scores
                    y_pred = torch.max(mix_outputs.data, 1)[1]
                    total += labels.size(0)
                    correct += (y_pred == labels).sum().item()
                    weighted_f1_train.append(f1_score(labels, y_pred, average = 'weighted'))
                    f1_train.append(f1_score(labels.cpu().detach().numpy(), y_pred.cpu().detach.numpy(), average = 'macro'))
                    
                else:
                    data = Variable(data.type(self.Tensor))
                    labels = Variable(labels.type(self.LongTensor))
                    data, labels = data.to(device), labels.to(device)
                    
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward + backward
                    _, outputs = self.model(data)
                    loss = self.criterion_cls(outputs, labels)
                    loss.backward()
                    
                    # Optimize
                    self.optimizer.step()
                    
                    # Recover accurate prediction and F1 score
                    y_pred = torch.max(outputs.data, 1)[1]
                    total += labels.size(0)
                    correct += (y_pred == labels).sum().item()
                    f1_train.append(f1_score(labels.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), average = 'macro'))
            
            # Recover accuracy and F1 score 
            train_acc = 100 * correct // total
            train_info[e]["Loss"] = loss.detach().numpy()
            train_info[e]["Accuracy"] = train_acc
            train_info[e]["F1_score"] = np.mean(f1_train)
                
            # Evaluate the model
            test_correct, test_total = 0,0
            weighted_f1_test, f1_test = [],[]
            Predictions = []
            Labels = []
            
            # Set evaluation mode
            self.model.eval()
            
            for j, (test_data, test_labels) in enumerate(self.validation_dataloader):

                # Recover data, labels
                test_data = Variable(test_data.type(self.Tensor))
                test_labels = Variable(test_labels.type(self.LongTensor))

                # Recover outputs
                _, test_outputs = self.model(test_data)
                test_loss = self.criterion_cls(test_outputs, test_labels)

                    # Recover accurate prediction and F1 scores
                test_y_pred = torch.max(test_outputs, 1)[1]
                test_total += test_labels.size(0)
                test_correct += (test_y_pred == test_labels).sum().item()
                f1_test.append(f1_score(test_labels.cpu().detach().numpy(), test_y_pred.cpu().detach().numpy(), average = 'macro'))

                # Recover labels and prediction
                Predictions.append(test_y_pred.detach().numpy())
                Labels.append(test_labels.detach().numpy())

            # Recover accuracy and F1 score
            test_acc = 100 * test_correct // test_total
            test_info[e]["Loss"] = test_loss.detach().numpy()
            test_info[e]["Accuracy"] = test_acc
            test_info[e]["F1_score"] = np.mean(f1_test)

            
            num+=1
            averAcc = averAcc + test_acc
            averF1 = averF1 + np.mean(f1_test)
            
            if test_acc > bestAcc:
                bestAcc = test_acc
                best_epochs_acc = e
                Y_true_acc = Predictions
                Y_pred_acc = Labels
                
            if np.mean(f1_test) > bestF1:
                bestF1 = np.mean(f1_test)
                best_epochs_F1 = e
                Y_true_F1 = Predictions
                Y_pred_F1 = Labels

        if num > 0:
            averAcc = averAcc / num
            averF1 = averF1 / num
            
        logger.info("Training finished")
        print('The average accuracy is:', averAcc)
        print('The best accuracy is:', bestAcc)
        print('The average F1 score is:', averF1)
        print('The best F1 score is:', bestF1)
        
        if save:
            logger.info("Saving config files")
            json.dump(config, open(config_path, 'w' ))
            
            logger.info("Saving parameters")
            torch.save(self.model.state_dict(), model_path)
            torch.save(self.optimizer.state_dict(), optimizer_path)
            
            print('Model state here:', model_path)
            print('Optimizer state here:', optimizer_path)
            print('Config file here:', config_path, '\n')
                
        return train_info, test_info, bestAcc, averAcc, bestF1, averF1, Y_true_acc, Y_pred_acc, Y_true_F1, Y_pred_F1
    
    
    def evaluate(self, config_path, model_path, optimizer_path, gpu_id):

        """
        Evaluate a model on test set.

        Args:
            config_path (str): Path to recover config dictionnary,
            model_path (str): Path to recover model hyperparameters,
            optimizer_path (str): Path to recover optimizer parameters.

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
        
        # Load model parameters
        model = Transformer(**model_config)
        model.load_state_dict(torch.load(model_path))
        
        # Move to gpu if available
        available, device = define_device(gpu_id)
        if available:
            if torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
        model.to(device)
        
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
                data = Variable(data.type(self.Tensor))
                labels = Variable(labels.type(self.LongTensor))
                data, labels = data.to(device), labels.to(device)
                _, outputs = model(data)
                predicted = torch.max(outputs.data, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                f1_test.append(f1_score(labels.cpu().detach().numpy(), predicted.cpu().detach().numpy(), average = 'macro'))

        accuracy = correct / total * 100
        F1_score = np.mean(f1_test)
        
        logger.info('Evaluation finished')
        print("Accuracy on test set: {} \nF1 score on test set: {}".format(accuracy, F1_score))
        
        return accuracy , F1_score
        

def main(): 
   
    """
    Train or evaluate model depending on args given in command line
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
        train_results = trans.train(config, model_path, optimizer_path, config_path, gpu_id = gpu_id, save = save)
        train_info, test_info, bestAcc, averAcc, bestF1, averF1, Y_true_acc, Y_pred_acc, Y_true_F1, Y_pred_F1 = train_results
                  
    # Evaluate model
    if Test_bool:
        accuracy, F1_score = trans.evaluate(config_path, model_path, optimizer_path, gpu_id)
        
        

if __name__ == "__main__":
    main()