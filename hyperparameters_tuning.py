#!/opt/anaconda3/bin/python

"""
This script is used to tune the hyperparameters of the model using Ray Tune.
Website: `<https://docs.ray.io/en/latest/index.html>`

Usage: type "from hyperparameter_tuning import <class>" to use one of its classes.
       type "from hyperparameter_tuning import <function>" to use one of its functions.

Contributors: Ambroise Odonnat.
"""

import os 
import json
import ray
import torch

import numpy as np

from os import listdir
from os.path import isfile, join
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from functools import partial
from torch.autograd import Variable

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from data import Data
from dataloader import train_test_dataset, get_dataloader
from utils import define_device
from parser import get_tune_parser
from Model import Transformer

from loguru import logger


def weight_reset(model):
    
    """
    Reset parameters of the model between folds in cross-validation process.

    Args:
        model: Deep Learning model (inherited from torch.nn.Modules).
    """
    
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            m.reset_parameters()
        
        
def train_validation(config, train_set, val_set, gpu_id, check = False):

    """
    Realise a cross-validation on the training set and send average validation loss to tune (imported from ray module)
    to help tuning hyperparameters.

    Args:
        config (dict): Dictionnary of dictionnary containing model hyperparamaters, optimizer parameters 
                       and training configuration --> dataloaders parameters (batch size, number of workers),
                       number of epochs, mix-up parameters,
        train_set (array): Training set,
        n_splits (int): Number of folds in the cross-validation process,
        check (bool): Save model and optimizer state during training.

    """

    # Recover model, optimizer and training configuration
    training_config , model_config, optimizer_config = config['Training'], config['Model'], config['Optimizer']
                                                             
    # Tensor format
    Tensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    
    # Define model
    model = Transformer (**model_config)

    # Move model to gpu if available
    available, device = define_device(gpu_id) 
    if available:
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(model)
    model.to(device)

    # Define loss
    if available:
        criterion_cls = torch.nn.CrossEntropyLoss().cuda()
    else:
        criterion_cls = torch.nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = optimizer_config['lr'],\
                                          betas=(optimizer_config['b1'], optimizer_config['b2']))
    
    # Recover all data and labels
    data_train, labels_train = train_set
    data_test, labels_test = val_set
         
    # Recover train and test loaders
    data_train = np.expand_dims(data_train, axis = 1)
    data_test = np.expand_dims(data_test, axis = 1)

    # Create dataloader
    batch_size, num_workers, balanced = training_config['batch_size'], training_config['num_workers'], training_config['balanced']
    train_dataloader = get_dataloader(data_train, labels_train, batch_size, num_workers, balanced)
    test_dataloader = get_dataloader(data_test, labels_test, batch_size, num_workers, False)

    # Reset the model parameters
    model.apply(weight_reset)

    # Recover number of epochs
    n_epochs = training_config['Epochs']

    # loop over the dataset n_epochs times
    mix_up = training_config["Mix-up"]
    for epoch in range(n_epochs):  

        # Set training mode
        model.train()

        # Train the model
        for i, (data, labels) in enumerate(train_dataloader):

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
                mix_data = Variable(mix_data.type(Tensor), requires_grad = True)
                data = Variable(data.type(Tensor), requires_grad = True)
                labels = Variable(labels.type(LongTensor))
                rolled_labels = Variable(rolled_labels.type(LongTensor))
                mix_data, labels, rolled_labels = mix_data.to(device), labels.to(device), rolled_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward
                _, mix_outputs = model(mix_data)
                loss = (lambdas.squeeze()).to(device)*criterion_cls(mix_outputs, labels) + (1-lambdas.squeeze()).to(device)*criterion_cls(mix_outputs, rolled_labels)
                loss = loss.sum()
                loss.backward()

                # Optimize
                optimizer.step()

            else:
                data = Variable(data.type(Tensor), requires_grad = True)
                labels = Variable(labels.type(LongTensor))
                data, labels = data.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward
                _, outputs = model(data)
                loss = criterion_cls(outputs, labels)
                loss.backward()

                # Optimize
                optimizer.step()

        # Validation
        test_loss = 0.0
        test_steps = 0
        test_total = 0
        test_correct = 0
        test_F1_score = 0.0

        # Set evaluation mode
        model.eval()

        for j, (test_data, test_labels) in enumerate(test_dataloader):

            # Recover data, labels
            test_data = Variable(test_data.type(Tensor), requires_grad = True)
            test_labels = Variable(test_labels.type(LongTensor))
            test_data, test_labels = test_data.to(device), test_labels.to(device)

            # Recover outputs
            _, test_outputs = model(test_data)
            loss = criterion_cls(test_outputs, test_labels)
            test_y_pred = torch.max(test_outputs, 1)[1]

            # Update val_loss, accuracy and F1_score
            test_loss += loss.cpu().detach().numpy()
            test_steps += 1  
            test_correct += (test_y_pred == test_labels).sum().item()
            test_total += test_labels.size(0)     
            test_F1_score += f1_score(test_labels.cpu().detach().numpy(), test_y_pred.cpu().detach().numpy(), average = 'macro')

        if check:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)   
        
        averLoss = (test_loss / test_steps)
        averAcc = (test_correct / test_total) 
        averF1 = (test_F1_score / test_steps)
        
        tune.report(loss = averLoss, accuracy = averAcc, F1_score = averF1)
    
    logger.info("Finished Training") 

    
def test_accuracy(config, model_state, test_set, gpu_id):
    
    """
    Evaluate a model on test set.

    Args:
        config (dict): Dictionnary of dictionnary containing model hyperparamaters, optimizer parameters 
                       and training configuration --> dataloaders parameters (batch size, number of workers),
                       number of epochs, mix-up parameters,
        model: Deep Learning model (inherited from torch.nn.Modules),
        test_set (array): Test set,
        device (str): Indicates the type of device used (gpu or cpu).

    Returns:
        tuple: accuracy (float): Average accuracy on the test set,
               F1_score (float): Average F1 score on the test set.
    """
    
    model = Transformer(**config['Model']) 
    
    # Move model to gpu if available
    available, device = define_device(gpu_id)
    if available:
        if torch.cuda.device_count() > 1:
            model = torch.nn.parallel.DistributedDataParallel(model)
    model.to(device)

    # Load model state
    model.load_state_dict(model_state)
    
    # Set evaluation mode 
    model.eval()
    
    # Recover test data and labels
    data_test, labels_test = test_set
    data_test = np.expand_dims(data_test, axis = 1)
    
    # Tensor format
    Tensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    
    # Create dataloader
    training_config = config['Training']
    batch_size, num_workers = training_config['batch_size'], training_config['num_workers']
    test_dataloader = get_dataloader(data_test, labels_test, batch_size, num_workers, False)
    
    correct = 0
    total = 0
    F1_score = 0.0
    steps = 0

    with torch.no_grad():
        for i, (data, labels) in enumerate(test_dataloader):
            data = Variable(data.type(Tensor), requires_grad = True)
            labels = Variable(labels.type(LongTensor))
            data, labels = data.to(device), labels.to(device)
            _, outputs = model(data)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            F1_score += f1_score(labels.cpu().detach().numpy(), predicted.cpu().detach().numpy(), average = 'macro')
            steps += 1
            
    logger.info("Finished evaluating")
    accuracy = (correct / total) 
    F1_score /= steps
    return accuracy, F1_score

def get_config(tuning_config_path):
    
    """
    Function to obtain the search domain for hyperparameters.
    ! Warning! slice_size must be a multiple of num_heads.

    Args:
        tuning_config (dict): Dictionnary of dictionnary containing in list format search model for model hyperparamaters, optimizer parameters 
                       and training configuration --> dataloaders parameters (batch size, number of workers),
                       number of epochs, mix-up parameters.

    Returns:
            tuning_config (dict): Dictionnary of dictionnary containing in categorical format search model hyperparamaters, optimizer parameters 
                    and training configuration --> dataloaders parameters (batch size, number of workers),
                    number of epochs, mix-up parameters.
    """
    
    # Recover tuning config dictionnary
    with open(tuning_config_path) as f:
        tuning_config = json.loads(f.read())
        
    config = tuning_config 
    config['Model']['attention_dropout'] = tune.choice(tuning_config['Model']['attention_dropout'])
    config['Model']['attention_negative_slope'] = tune.choice(tuning_config['Model']['attention_negative_slope'])
    config['Model']['attention_kernel_size'] = tune.choice(tuning_config['Model']['attention_kernel_size'])
    config['Model']['attention_stride'] = tune.choice(tuning_config['Model']['attention_stride'])
    config['Model']['spatial_dropout'] = tune.choice(tuning_config['Model']['spatial_dropout'])
    config['Model']['position_kernel_size'] = tune.choice(tuning_config['Model']['position_kernel_size'])
    config['Model']['emb_negative_slope'] = tune.choice(tuning_config['Model']['emb_negative_slope'])
    config['Model']['time_kernel_size'] = tune.choice(tuning_config['Model']['time_kernel_size'])
    config['Model']['time_stride'] = tune.choice(tuning_config['Model']['time_stride'])
    config['Model']['slice_size'] = tune.choice(tuning_config['Model']['slice_size']) # A multiple of each num_heads possible value
    config['Model']['depth'] = tune.choice(tuning_config['Model']['depth'])
    config['Model']['num_heads'] = tune.choice(tuning_config['Model']['num_heads']) # A divider of each slice_size possible value
    config['Model']['transformer_dropout'] = tune.choice(tuning_config['Model']['transformer_dropout'])
    config['Model']['forward_expansion'] = tune.choice(tuning_config['Model']['forward_expansion'])
    config['Model']['forward_dropout'] = tune.choice(tuning_config['Model']['forward_dropout'])
    
    config['Optimizer']['lr'] = tune.choice(tuning_config['Optimizer']['lr'])
    config['Optimizer']['b1'] = tune.choice(tuning_config['Optimizer']['b1'])
    config['Optimizer']['b2'] = tune.choice(tuning_config['Optimizer']['b2'])
    
    config['Training']['batch_size'] = tune.choice(tuning_config['Training']['batch_size'])
    config['Training']['BETA'] = tune.choice(tuning_config['Training']['BETA'])
    
    return config


def main(validation_size = 0.15):

    """
    Use Ray Tune to tune hyperparameters.
    ! Warning ! Trials seems to crash if cpu and gpu resources are not the same in the tune.run() function.
    
    Args:
        validation_size (float): Size of the validation set.

    Returns:
        tuple: best_trial.config (dict): Dictionnary of dictionnary containing best model hyperparamaters, optimizer parameters 
                                         and training configuration --> dataloaders parameters (batch size, number of workers),
                                         number of epochs, mix-up parameters,
               model_state (dict): Best model state,
               optimizer_state (dict): Best optimizer for best model.
    """

    parser = get_tune_parser()
    args = parser.parse_args()
    
    # Recover data path
    data_path = args.path_data
    data_config_path = args.path_config_data
    tuning_config_path = args.path_tuning_config
    results_path = args.path_results
    best_states_path = args.path_best_states
    config_path = args.path_best_config

    # Recover gpu_id, resources, num_samples and max_num_epochs
    gpu_id = args.gpu_id  
    num_samples = args.n_samples
    max_num_epochs = args.max_n_epochs
    gpu_resources = args.gpu_resources
    cpu_resources = args.cpu_resources
    
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
    random_state = data_config['random_state']
    shuffle = data_config['shuffle']
    
    # Recover data
    dataset = Data(folder,channel_fname,wanted_event_label, list_channel_type,binary_classification, selected_rows)
    allData, allLabels, allSpikeTimePoints, allTimes = dataset.csp_data()

    # Split data and labels in train, validation and test sets
    data_train, labels_train, data_test, labels_test = train_test_dataset(allData, allLabels,\
                                                               train_size, shuffle, random_state)
    new_train_size = 1 - validation_size/train_size
    data_train, labels_train, data_val, labels_val = train_test_dataset(data_train, labels_train,\
                                                               new_train_size, shuffle, random_state)
    train_set = (data_train, labels_train)
    val_set = (data_val, labels_val)
    test_set = (data_test, labels_test)
    
    # Recover tuning config dictionnary
    config = get_config(tuning_config_path)
    
    scheduler = ASHAScheduler(
        metric = "loss",
        mode = "min",
        max_t = max_num_epochs,
        grace_period = 1,
        reduction_factor = 2)
    
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "F1_score", "training_iteration"])
    
    # Tune hyperparameters
    result = tune.run(
        partial(train_validation, train_set = train_set, val_set = val_set, gpu_id = gpu_id, check = True),
        resources_per_trial={"cpu": cpu_resources, "gpu": gpu_resources},
        config = config,
        num_samples = num_samples,
        scheduler = scheduler,
        progress_reporter = reporter,
        local_dir = results_path)

    logger.info('Done')
    best_trial = result.get_best_trial("loss", "min", "all")
    print("Best trial config: {}".format(
        best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best trial final validation F1 score: {}".format(
        best_trial.last_result["F1_score"]))
    
    # Recover model and optimizer parameters
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_model_config = best_trial.config

    # Evaluate the best model on the test set
    test_acc, test_F1_score = test_accuracy(best_model_config, model_state, test_set, gpu_id)
    
    print("Best trial test set accuracy: {}, best trial test set F1 score {}".format(test_acc,test_F1_score))

    # Saving best configuration file
    json.dump(best_trial.config, open(config_path, 'w' ))
    print('Best Config file here:', config_path)
    
    # Saving best model and optimizer states
    torch.save((model_state, optimizer_state), best_states_path)
    print('Model and optimizer states here:', best_states_path)
    
    
    return best_model_config, model_state, optimizer_state



if __name__ == "__main__":
    main()
