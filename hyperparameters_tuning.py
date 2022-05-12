#!/usr/bin/env python

"""
This script is used to tune the hyperparameters of the model using Ray Tune.
Website: `<https://docs.ray.io/en/latest/index.html>`_.

Usage: type "from hyperparameters_tuning import <function>" to use one of its functions.

Contributors: Ambroise Odonnat.
"""

import json
import ray
import torch

import numpy as np

from functools import partial
from loguru import logger
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from torch.autograd import Variable
from torchmetrics.functional import  accuracy, specificity, precision_recall, f1_score 

from custom_losses import get_classification_loss, get_detection_loss
from data import Data
from dataloader import get_dataloader
from early_stopping import EarlyStopping
from learning_rate_warmup import NoamOpt
from models import ClassificationBertMEEG, DetectionBertMEEG
from utils import define_device
        
          
def train_validation(task, config, train_set, val_set, n_classes, cost_sensitive, lambd,
                     weight_decay, amsgrad, scheduler, warmup, gpu_id, check=False):

    """
    Train the model corresponding to task.
    Tune hyperparameters thanks to Ray Tune `<https://docs.ray.io/en/latest/index.html>`_.

    Args:
        task (str): Indicates the model to use.
        config (dict): Dictionnary containing training configuration, model and optimizer hyperparameters.
        train_set (array): Training set with data, labels and spike events times.
        val_set (array): Validation set with data, labels and spike events times.
        n_classes (int): Number of classes in the dataset.
        cost_sensitive (bool): If True, use cost-sensitive loss.
        lambd (float): Modulate the influence of the cost-sensitive weight.
        weight_decay (float): Value of weight decay in optimizer.
        amsgrad (bool): Use AMSGrad variant of Adam.
        scheduler (bool): Use ReduceLROnPLateau scheduler.
        warmup (int): If warmup > 0, apply warm-up steps on the learning rate.
        gpu_id (int): Index of cuda device to use if available.
        check (bool): If True, save model state during training.

    """
    
    # Tensor format
    Tensor = torch.FloatTensor
    LongTensor = torch.LongTensor
            
    # Recover model, optimizer and training configuration
    training_config , model_config, optimizer_config = config['Training'], config['Model'], config['Optimizer']

    # Recover training parameters
    batch_size, num_workers = training_config['batch_size'], training_config['num_workers']
    n_epochs, final_epoch = training_config['Epochs'], training_config['Epochs']
    mix_up, BETA = training_config["Mix-up"], training_config["BETA"]
    
    train_data, train_labels, train_spike_events = train_set
    val_data, val_labels, val_spike_events = val_set
    
    if task == 'spike_counting':
        
        # Create dataloader
        train_dataloader = get_dataloader(train_data, train_labels, batch_size, True, num_workers)
        validation_dataloader = get_dataloader(val_data, val_labels, batch_size, False, num_workers) 

        # Define model
        model = ClassificationBertMEEG(**model_config)
        
        # Define losses
        train_criterion_cls = get_classification_loss(n_classes, cost_sensitive, lambd)
        val_criterion_cls = torch.nn.CrossEntropyLoss()
    
    elif task == 'spike_detection':
    
        # Create dataloader
        train_dataloader = get_dataloader(train_data, train_spike_events, batch_size, True, num_workers)
        validation_dataloader = get_dataloader(val_data, val_spike_events, batch_size, False, num_workers) 

        # Define model
        n_time_windows = model_config['n_time_windows']
        model = DetectionBertMEEG(**model_config)

        # Define losses
        train_criterion_cls = get_detection_loss(cost_sensitive, lambd)
        val_criterion_cls = torch.nn.CrossEntropyLoss()
        
    # Move to gpu if available
    available, device = define_device(gpu_id)
    if available:
        train_criterion_cls = train_criterion_cls.cuda()
        val_criterion_cls = val_criterion_cls.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    model.to(device)

    # Define optimizer
    lr, b1, b2 = optimizer_config['lr'], optimizer_config['b1'], optimizer_config['b2']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(b1, b2),
                                      weight_decay=weight_decay, amsgrad=amsgrad)

    # Define warmup method
    if warmup:
        warmup_scheduler = NoamOpt(lr, warmup, optimizer) 

    # Define scheduler
    self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5,
                                                                   patience=20, min_lr=1e-5, verbose=False)

    # Loop over the dataset n_epochs times
    for e in range(n_epochs):

        # Set training mode
        model.train()
        for i, (data, labels) in enumerate(self.train_dataloader):

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
                mix_data = Variable(mix_data.type(Tensor), requires_grad=True)
                data = Variable(data.type(Tensor), requires_grad=True)
                labels = Variable(labels.type(LongTensor))
                rolled_labels = Variable(rolled_labels.type(LongTensor))
                mix_data, labels, rolled_labels = mix_data.to(device), labels.to(device), rolled_labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                _, mix_outputs = model(mix_data)

                if task == 'spike_counting':
                    
                    # Compute mix-up loss
                    lambdas = lambdas.squeeze().to(device)
                    loss = (lambdas * train_criterion_cls(mix_outputs, labels) 
                            + ((1-lambdas) * train_criterion_cls(mix_outputs, rolled_labels)))
                elif task == 'spike_detection':
                    
                    # Concatenate the batches
                    stack_mix_outputs = rearrange(mix_outputs, '(b) (v) o -> (b v) o') 
                    stack_labels = rearrange(labels, '(b) (v) -> (b v)') 
                    stack_rolled_labels = rearrange(rolled_labels, '(b) (v) -> (b v)') 
                    
                    # Compute mix-up loss
                    lambdas = lambdas.squeeze().to(device)
                    loss = (lambdas * train_criterion_cls(stack_mix_outputs, stack_labels) 
                            + ((1-lambdas) * train_criterion_cls(stack_mix_outputs, stack_rolled_labels)))
                loss = loss.sum()

                # Backward
                loss.backward()

                # Update learning rate
                if warmup:
                    warmup_scheduler.step()
                else:
                    optimizer.step()

            else:

                # Format conversion and move to device
                data = Variable(data.type(Tensor), requires_grad=True)
                labels = Variable(labels.type(LongTensor))
                data, labels = data.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                _, outputs = model(data)
                y_pred = torch.max(outputs.data, -1)[1] 

                if task == 'spike_counting':
                    
                    # Compute training loss
                    loss = train_criterion_cls(outputs, labels)
                elif task == 'spike_detection':
                    
                    # Concatenate the batches
                    stack_outputs = rearrange(outputs, '(b) (v) o -> (b v) o') 
                    stack_labels = rearrange(labels, '(b) (v) -> (b v)') 
                    
                    # Compute training loss
                    loss = train_criterion_cls(stack_outputs, stack_labels)

                # Backward
                loss.backward()

                # Update learning rate
                if warmup:
                    warmup_scheduler.step()
                else:
                    optimizer.step()

        # Evaluate the model
        val_loss = 0
        val_steps = 0

        # Set evaluation mode
        self.model.eval()
        for j, (val_data, val_labels) in enumerate(self.validation_dataloader):

            # Format conversion and move to device
            val_data = Variable(val_data.type(Tensor), requires_grad=True)
            val_labels = Variable(val_labels.type(LongTensor))
            val_data, val_labels = val_data.to(device), val_labels.to(device)

            # Forward
            _, val_outputs = model(val_data)

            if task == 'spike_counting':
                
                # Compute validation loss
                loss = val_criterion_cls(val_outputs, val_labels)
            elif task == 'spike_detection':
                
                # Compute validation loss
                loss = self.val_criterion_cls(stack_val_outputs, stack_val_labels)

            # Detach from device
            loss = loss.cpu().detach().numpy()

            # Update learning rate if validation loss does not improve
            if scheduler:
                lr_scheduler.step(loss)

            # Recover training loss and metrics
            val_loss += loss 
            val_steps += 1
                
        if check:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)   
        
        loss /= val_steps
        tune.report(loss=loss)
    
    logger.info("Finished Training") 

    
def test(config, test_set, model_state, gpu_id):
    
    """
    Evaluate a model on test set.

    Args:
        config (dict): Dictionnary containing training_configuration, model and optimizer hyperparameters.
        train_set (array): Training set with data, labels and spike events times.
        model: Model inherited from torch.nn.Modules.
        gpu_id (int): Index of cuda device to use if available.
        
    """
    # Tensor format
    Tensor = torch.FloatTensor
    LongTensor = torch.LongTensor
            
    # Recover model, optimizer and training configuration
    training_config , model_config, optimizer_config = config['Training'], config['Model'], config['Optimizer']

    # Recover training parameters
    batch_size, num_workers = training_config['batch_size'], training_config['num_workers']
    test_data, test_labels, test_spike_events = test_set
    
    if task == 'spike_counting':
        
        # Create dataloader
        test_dataloader = get_dataloader(test_data, test_labels, batch_size, True, num_workers)

        # Define model
        model = ClassificationBertMEEG(**model_config)

    elif task == 'spike_detection':
    
        # Create dataloader
        test_dataloader = get_dataloader(test_data, test_spike_events, batch_size, True, num_workers)

        # Define model
        n_time_windows = model_config['n_time_windows']
        model = DetectionBertMEEG(**model_config)

    # Move model to gpu if available
    available, device = define_device(gpu_id)
    if available:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    model.to(device)

    # Load model state
    model.load_state_dict(model_state)
    
    # Set evaluation mode 
    model.eval()

    if task == 'spike_counting':

        # Initialize metrics
        Accuracy, Specificity, Recall, Precision, F1_score = 0, 0, 0, 0, 0
        steps = 0
        
        with torch.no_grad():
            for i, (data, labels) in enumerate(self.test_dataloader):
                
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
                
                # Recover metrics
                Accuracy += accuracy(pred, labels, average='micro').cpu().detach().item()
                Specificity += specificity(pred, labels, average='macro', num_classes=self.n_classes).cpu().detach().item()
                precision, recall = precision_recall(pred, labels, average='macro', num_classes=self.n_classes)
                Recall += recall.cpu().detach().item()
                Precision += precision.cpu().detach().item()
                F1_score += f1_score(pred, labels, average='macro', num_classes=self.n_classes).cpu().detach().item()
                steps += 1

            Accuracy /= steps
            Specificity /= steps
            Recall /= steps
            Precision /= steps
            F1_score /= steps
        
        logger.info('Evaluation finished')
        
        print('Average Accuracy on test set:', "%.4f" % round(Accuracy, 4))
        print('Average Specificity on test set:', "%.4f" % round(Specificity, 4))
        print('Average Recall on test set:', "%.4f" % round(Recall, 4))
        print('Average Precision on test set:', "%.4f" % round(Precision, 4))
        print('Average F1 score on test set:', "%.4f" % round(F1_score, 4))
        
    elif task == 'spike_detection':
        
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
        
        logger.info('Evaluation finished')
        
        print('Average Accuracy on test set:', "%.4f" % round(Accuracy, 4))
        print('Average Specificity on test set:', "%.4f" % round(Specificity, 4))
        print('Average Sensitivity on test set:', "%.4f" % round(Sensitivity, 4))
        print('Average Precision on test set:', "%.4f" % round(Precision, 4))
        print('Average F1 score on test set:', "%.4f" % round(F1_score, 4))

        

def get_tuning_config(tuning_config):
    
    """
    Function to obtain the search domain for hyperparameters.
    `<num_heads>` must be a dividor of `<emb_size>`.

    Args:
        tuning_config (dict): Dictionnary containing search domain in list format 
                              of training configuration, model and optimizer hyperparameters.

    Returns:
        tuning_config (dict): Dictionnary containing search domain in categorical format
                              of training configuration, model and optimizer hyperparameters.
    """
    
    config = tuning_config.copy()
    
    config['Model']['attention_dropout'] = tune.choice(tuning_config['Model']['attention_dropout'])
    config['Model']['attention_kernel'] = tune.choice(tuning_config['Model']['attention_kernel'])
    config['Model']['attention_stride'] = tune.choice(tuning_config['Model']['attention_stride'])
    config['Model']['spatial_dropout'] = tune.choice(tuning_config['Model']['spatial_dropout'])
    config['Model']['position_kernel'] = tune.choice(tuning_config['Model']['position_kernel'])
    config['Model']['time_kernel'] = tune.choice(tuning_config['Model']['time_kernel'])
    config['Model']['time_stride'] = tune.choice(tuning_config['Model']['time_stride'])
    config['Model']['emb_size'] = tune.choice(tuning_config['Model']['emb_size']) 
    config['Model']['depth'] = tune.choice(tuning_config['Model']['depth'])
    config['Model']['num_heads'] = tune.choice(tuning_config['Model']['num_heads']) 
    config['Model']['transformer_dropout'] = tune.choice(tuning_config['Model']['transformer_dropout'])
    config['Model']['forward_expansion'] = tune.choice(tuning_config['Model']['forward_expansion'])
    config['Model']['transformer_dropout'] = tune.choice(tuning_config['Model']['transformer_dropout'])

    return config


from os import listdir
from os.path import isfile, join
from parser import get_tune_parser


def main(validation_size = 0.15):

    """
    Use Ray Tune to tune hyperparameters: save the best configuration file with corresponding model and optimizer parameters.
    Configurations files must be saved in .json format.
    ! *** Warning --> If cuda is available, trials seems to crash if cpu and gpu resources are not the same in the tune.run() function *** !
    
    Args:
        validation_size (float): Size of the validation set.

    Tuning:
            To tune hyperparameters of the model, run the following command in your terminal:
            
            python hyperparameters_tuning.py --tune --path-data [path to the data] --path-config_data 
            [path to configuration file for data] --path-tuning_config [path to configuration file for tuning] --path-results 
            [path to save search results] --path-best_states [path to save bets model and optimizer parameters] --path-best_config 
            [path to save best corresponding configuration file] -gpu [index of the cuda device to use if available] --n_samples [number of samples in Ray]
            --max_n_epochs [number of maxima epochs for each trial] --gpu_resources [gpu resources] --cpu_resources [cpu resources]
            --metric [metric to tune on ("loss" for instance)] --mode [mode to tune on ("min" for instance)]
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
    
    # Recover metric and mode for Ray tune
    metric = args.metric
    mode = args.mode
    
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
    with open(tuning_config_path) as f:
        tuning_config = json.loads(f.read())
        
    # Recover tuning config dictionnary
    config = get_config(tuning_config)
    
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
    best_trial = result.get_best_trial(metric, mode, "all")
    print("Best trial config: {}".format(
        best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    print("Best trial final validation F1 score: {}".format(
        best_trial.last_result["F1_score"]))
    print("Best trial epochs: {}".format(
        best_trial.last_result["training_iteration"]))
    
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

    

if __name__ == "__main__":
    main()
