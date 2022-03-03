#!/opt/anaconda3/bin/python

"""
This script is used to tune the hyperparameters of the model using Ray Tune.
Website: `<https://docs.ray.io/en/latest/index.html>`

Usage: type "from hyperparameter_tuning import <class>" to use one of its classes.
       type "from hyperparameter_tuning import <function>" to use one of its functions.

Contributors: Ambroise Odonnat.
"""

import os 
import ray
import torch



import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from functools import partial
from torch.autograd import Variable

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from dataloader import get_dataloader
from Model import Transformer


def weight_reset(model):
    
    """
    Reset parameters of the model between folds in cross-validation process.

    Args:
        model: Deep Learning model (inherited from torch.nn.Modules).
    """
    
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
            m.reset_parameters()
        
        
def cross_validation(config, train_set, n_splits, check = False):

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
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    # Define loss
    criterion_cls = torch.nn.CrossEntropyLoss()
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = optimizer_config['lr'],\
                                          betas=(optimizer_config['b1'], optimizer_config['b2']))
    
    # Recover all data and labels
    allData, allLabels = train_set
    
    # Split the train_set into k parts to do k-fold cross-validation
    kfold = KFold(n_splits = n_splits)

    # Initialize validation loss, accuracy and F1 score
    averLoss = 0.0
    averAcc = 0.0
    averF1 = 0.0
    
    
    for fold, (train_index, test_index) in enumerate(kfold.split(allData)):
        
        # Recover train and test loaders
        data_train, labels_train = allData[train_index], allLabels[train_index]
        data_test, labels_test = allData[test_index], allLabels[test_index]
        
        data_train = np.expand_dims(data_train, axis = 1)
        data_test = np.expand_dims(data_test, axis = 1)

        # Create dataloader
        batch_size, num_workers, balanced = training_config['batch_size'], training_config['num_workers'], training_config['balanced']
        train_dataloader = get_dataloader(data_train, labels_train, batch_size, num_workers, balanced)
        test_dataloader = get_dataloader(data_train, labels_train, batch_size, num_workers, balanced)

        # Reset the model parameters
        model.apply(weight_reset)
        
        # Recover number of epochs
        n_epochs = training_config['Epochs']
        
        # loop over the dataset n_epochs times
        mix_up = training_config["Mix-up"]
        for epoch in range(n_epochs):  
            running_loss = 0.0
            epoch_steps = 0
            
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
                    mix_data = Variable(mix_data.type(Tensor))
                    data = Variable(data.type(Tensor))
                    labels = Variable(labels.type(LongTensor))
                    rolled_labels = Variable(rolled_labels.type(LongTensor))
                    mix_data, labels, rolled_labels = mix_data.to(device), labels.to(device), rolled_labels.to(device)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward
                    _, mix_outputs = model(mix_data)
                    loss = lambdas.squeeze()*criterion_cls(mix_outputs, labels) + (1-lambdas.squeeze())*criterion_cls(mix_outputs, rolled_labels)
                    loss = loss.sum()
                    loss.backward()
                    
                    # Optimize
                    optimizer.step()
                    
                    # Update running_loss
                    running_loss += loss.item()
                    epoch_steps += 1
                    
                else:
                    data = Variable(data.type(Tensor))
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
                    
                    # Update running_loss
                    running_loss += loss.item()
                    epoch_steps += 1

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
                test_data = Variable(test_data.type(Tensor))
                test_labels = Variable(test_labels.type(LongTensor))

                # Recover outputs
                _, test_outputs = model(test_data)
                loss = criterion_cls(test_outputs, test_labels)
                test_y_pred = torch.max(test_outputs, 1)[1]
                
                # Update val_loss, accuracy and F1_score
                test_loss += loss.detach().numpy()
                test_steps += 1            
                test_total += test_labels.size(0)
                test_correct += (test_y_pred == test_labels).sum().item()
                test_F1_score += f1_score(test_labels, test_y_pred, average = 'macro')

            if check:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)
        
        
        
        averLoss += (test_loss / test_steps)
        averAcc += (test_correct / test_total) * 100
        averF1 += (test_F1_score / test_steps)

    tune.report(loss = (averLoss / n_splits), accuracy = (averAcc / n_splits), F1_score = (averF1 / n_splits))
    print("Finished Training") 

    
def test_accuracy(config, model, test_set, device="cpu"):
    
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
    for _ in range(20):
        with torch.no_grad():
            for i, (data, labels) in enumerate(test_dataloader):
                data = Variable(data.type(Tensor))
                labels = Variable(labels.type(LongTensor))
                data, labels = data.to(device), labels.to(device)
                _, outputs = model(data)
                predicted = torch.max(outputs, 1)[1]
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                F1_score += f1_score(labels, predicted, average = 'macro')
                steps += 1
            
    accuracy = (correct / total) * 100
    F1_score /= steps
    return accuracy, F1_score


def main(config, train_set, test_set, n_splits, results_path,num_samples = 10,\
         max_num_epochs = 10, gpus_per_trial = 10):

    """
    Use Ray Tune to tune hyperparameters.

    Args:
        config (dict): Dictionnary of dictionnary containing model hyperparamaters, optimizer parameters 
                       and training configuration --> dataloaders parameters (batch size, number of workers),
                       number of epochs, mix-up parameters,
        train_set (array): Training set,
        test_set (array): Test set,
        n_splits (int): Number of folds in the cross-validation process,
        results_path (str): Path to save training information (model and optimizer states, hyperparameters, results),
        num_samples (int): Number of data samples during hyperparameter search,
        max_num_epochs (int): Max time units per trial (in the scheduler),
        gpus_per_trial (int): Number of gpu to use per trial.

    Returns:
        tuple: df (dataframe): summary of results and hyperparameters on each of the num_samples sample,
               best_trial.config (dict): Dictionnary of dictionnary containing best model hyperparamaters, optimizer parameters 
                                         and training configuration --> dataloaders parameters (batch size, number of workers),
                                         number of epochs, mix-up parameters,
               model_state (dict): Best model state,
               optimizer_state (dict): Best optimizer for best model.
    """

    scheduler = ASHAScheduler(
        metric = "loss",
        mode = "min",
        max_t = max_num_epochs,
        grace_period = 1,
        reduction_factor = 2)
    
    reporter = CLIReporter(metric_columns=["loss", "accuracy", "F1_score", "training_iteration"])
    
    # Tune hyperparameters
    result = tune.run(
        partial(cross_validation, train_set = train_set, n_splits = n_splits, check = True),
        resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial},
        config = config,
        num_samples = num_samples,
        scheduler = scheduler,
        progress_reporter = reporter,
        local_dir = results_path)

    print('done')
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(
        best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best trial final validation F1 score: {}".format(
        best_trial.last_result["F1_score"]))
    
    best_model_config = best_trial.config['Model']
    best_trained_model = Transformer(**best_model_config) 
    
    # Move model to gpu if available
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    # Recover model and optimizer parameters
    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    # Evaluate the best model on the test set
    test_acc, test_F1_score = test_accuracy(best_trial.config, best_trained_model, test_set, device)
    print("Best trial test set accuracy: {}, best trial test set F1 score {}".format(test_acc,test_F1_score))

    df = result.dataframe()
    
    return df, best_trial.config, model_state, optimizer_state


""""
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
"""