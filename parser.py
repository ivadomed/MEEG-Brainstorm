#!/opt/anaconda3/bin/python

"""
This script is used to parse command line arguments. 

Usage: type "from parser import <function>" to use one of its functions.
       
Contributors: Ambroise Odonnat.
"""

import argparse
import os 


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError("readable_dir: {} is not a valid direction path".format(path))

    
def file_path(path):
    return path
    
    
def get_parser():
    
    parser = argparse.ArgumentParser(
        prog = os.path.basename(__file__).strip('.py'),
        description = 'Get command instruction and configuration file',
        add_help = True  
    )
     
    # COMMAND ARGUMENTS

    command_group = parser.add_mutually_exclusive_group(required = True)

    
    command_group.add_argument('--train', dest = 'train', action = 'store_true',
                               help = 'Perform training on data.')
    
    command_group.add_argument('--test', dest = 'test', action = 'store_true',
                               help = 'Perform testing on trained model.')
    
    
    # ARGUMENTS
    
    parser.add_argument('-pd', '--path-data', dest = 'path_data', required = True, type = dir_path,
                               help = 'Path to data in .mat format')
    
    parser.add_argument('-pcd', '--path-config_data', dest = 'path_config_data', required = True, type = file_path,
                               help = 'Path to configuration dictionnary for data.')   
    
    parser.add_argument('-pct', '--path-config_training', dest = 'path_config_training', required = True, type = file_path,
                               help = 'Path to configuration dictionnary for training.')
    
    parser.add_argument('-pm', '--path-model', dest = 'path_model', required = True, type = file_path,
                               help = 'Path to save model state.')
    
    parser.add_argument('-pc', '--path-config', dest = 'path_config', required = True, type = file_path,
                               help = 'Path to save training configuration file.')
    
    parser.add_argument('-gpu', '--gpu_id', dest = 'gpu_id', required = True, type = int,
                               help = 'Id of the wanted gpu device.') 
    

    # OPTIONAL ARGUMENTS
    
    optional_args = parser.add_argument_group('OPTIONAL ARGUMENTS')
   
    optional_args.add_argument('--save', dest = 'save', action = 'store_true', required = False,
                               help = 'Save model and optimizer states as well as training configuration file.')
    
    optional_args.add_argument('--scheduler', dest = 'scheduler', action = 'store_true', required = False,
                               help = 'Use a scheduler on the learning rate.')
    
    optional_args.add_argument('--amsgrad', dest = 'amsgrad', action = 'store_true', required = False,
                               help = 'Use AMSGrad instead of ADAM as optimizer algorithm.')
    
    optional_args.add_argument('-weight_decay', dest = 'weight_decay', required = False, type = float,
                               help = 'Weight_decay in optimizer for L2 regularization.')

    optional_args.add_argument('-l1_weight', dest = 'l1_weight', required = False, type = float,
                               help = 'Weight for L1 regularization.')

    optional_args.add_argument('-l2_weight', dest = 'l2_weight', required = False, type = float,
                               help = 'Weight for L2 regularization.')
    
    return parser


def get_tune_parser():
    
    parser = argparse.ArgumentParser(
        prog = os.path.basename(__file__).strip('.py'),
        description = 'Get command instruction and configuration file',
        add_help = True  
    )

    # COMMAND ARGUMENTS
    
    command_group = parser.add_mutually_exclusive_group(required = False)

    
    command_group.add_argument('--tune', dest = 'tune', action = 'store_true',
                               help = 'Tune hyperparameters on model.')
    
    

    # ARGUMENTS

    parser.add_argument('-pd', '--path-data', dest = 'path_data', required = True, type = dir_path,
                               help = 'Path to data in .mat format')
    
    parser.add_argument('-pcd', '--path-config_data', dest = 'path_config_data', required = True, type = file_path,
                               help = 'Path to configuration dictionnary for data.')   

    parser.add_argument('-ptc', '--path-tuning_config', dest = 'path_tuning_config', required = True, type = file_path,
                               help = 'Path to tuning configuration dictionnary for hyperparameters tuning.') 

    parser.add_argument('-r', '--path-results', dest = 'path_results', required = True, type = dir_path,
                               help = 'Path to folder results.')
    
    parser.add_argument('-pbm', '--path-best_states', dest = 'path_best_states', required = True, type = file_path,
                               help = 'Path to save best model and optimizer states.')
    
    parser.add_argument('-pc', '--path-best_config', dest = 'path_best_config', required = True, type = file_path,
                               help = 'Path to save best model configuration file.')

    parser.add_argument('-n', '--n_samples', dest = 'n_samples', required = True, type = int,
                               help = 'Number of samples per trial during hyperparameters search.')
    
    parser.add_argument('-me', '--max_n_epochs', dest = 'max_n_epochs', required = True, type = int,
                               help = 'Maximum number of iterations per trial during hyperparameters search.')
    
    parser.add_argument('-gpu', '--gpu_id', dest = 'gpu_id', required = True, type = int,
                               help = 'Id of the wanted gpu device.')
    
    parser.add_argument('-gpu_res', '--gpu_resources', dest = 'gpu_resources', required = True, type = int,
                               help = 'Number of gpus to used per trial.')
    
    parser.add_argument('-cpu_res', '--cpu_resources', dest = 'cpu_resources', required = True, type = int,
                               help = 'Number of cpus to used per trial.')
    
    parser.add_argument('-m', '--metric', dest = 'metric', required = True, type = str,
                               help = 'Metric to tune hyperparameters in Ray tune.')
    
    parser.add_argument('-mo', '--mode', dest = 'mode', required = True, type = str,
                               help = 'Mode to tune hyperparameters in Ray tune.')
    
    return parser