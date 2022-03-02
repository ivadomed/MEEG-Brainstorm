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
    if isinstance(model, torch.nn.Conv2d) or isinstance(model, torch.nn.Linear):
        model.reset_parameters()
        
def cross_validation(model_config, optimizer_config, train_set, n_split, batch_size,\
                num_workers, balanced, n_epochs, mix_up, BETA):

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

    """
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    """
    
    # Recover all data and labels
    allData, allLabels = train_set
    
    # Split the train_set into k parts to do k-fold cross-validation
    kfold = KFold(n_splits=2)

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
        
        train_dataloader = get_dataloader(data_train, labels_train, batch_size, num_workers, balanced)
        test_dataloader = get_dataloader(data_train, labels_train, batch_size, num_workers, balanced)

        # Reset the model parameters
        model.apply(weight_reset)
        
        # loop over the dataset n_epochs times
        for epoch in range(n_epochs):  
            model.train()
            running_loss = 0.0
            epoch_steps = 0
            
            # Train the model
            for i, (data, labels) in enumerate(train_dataloader):
                
                if mix_up:
                    
                    # Apply a mix-up strategy for data augmentation as adviced here '<https://forums.fast.ai/t/mixup-data-augmentation/22764>'

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
            model.eval()
            test_loss = 0.0
            test_steps = 0
            test_total = 0
            test_correct = 0
            test_F1_score = 0.0
            
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

            
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
        
            
        averLoss += (test_loss / test_steps)
        averAcc += (test_correct / test_total) * 100
        averF1 += (test_F1_score / test_steps)

        tune.report(loss = (averLoss / n_split), accuracy = (averAcc / n_split), F1_score = (averF1 / n_split))
        print("Finished Training k-fold: %s "%fold)
    
    
def test_accuracy(model, test_set, batch_size, num_workers, device="cpu"):
    
    # Recover test data and labels
    data_test, labels_test = test_set
    data_test = np.expand_dims(data_test, axis = 1)
    
    # Tensor format
    Tensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    
    test_dataloader = get_dataloader(data_test, labels_test, batch_size, num_workers, False)
    correct = 0
    total = 0
    F1_score = 0.0
    steps = 0
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


def main(config, train_set, test_set, optimizer_config, n_split, batch_size,
         num_workers, balanced, n_epochs, mix_up, BETA,\
         num_samples = 10, max_num_epochs = 10, gpus_per_trial = 10):
    
    scheduler = ASHAScheduler(
        metric = "loss",
        mode = "min",
        max_t = max_num_epochs,
        grace_period = 1,
        reduction_factor = 2)
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "F1_score", "training_iteration"])
    
    result = tune.run(
        partial(cross_validation, optimizer_config = optimizer_config, train_set = train_set, n_split = n_split,\
                batch_size = batch_size, num_workers = num_workers, balanced = balanced,\
                n_epochs = n_epochs, mix_up = mix_up, BETA = BETA),
        resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial},
        config = config,
        num_samples = num_samples,
        scheduler = scheduler,
        progress_reporter = reporter,
        local_dir = "check_results")

    print('done')
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Transformer(**best_trial.config) 
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc, test_F1_score = test_accuracy(best_trained_model, test_set, batch_size, num_workers, device)
    print("Best trial test set accuracy: {}, best trial test set F1 score {}".format(test_acc,test_F1_score))


""""
if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=10, gpus_per_trial=0)
"""