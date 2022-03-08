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

    command_group = parser.add_mutually_exclusive_group(required = False)

    
    command_group.add_argument('--train', dest = 'train', action = 'store_true',
                               help = 'Perform training on data.')
    
    command_group.add_argument('--test', dest = 'test', action = 'store_true',
                               help = 'Perform testing on trained model.')
    
    parser.add_argument('--save', dest = 'save', action = 'store_true',
                               help = 'Save model and optimizer states as well as training configuration file.')

    # OPTIONAL ARGUMENTS
    optional_args = parser.add_argument_group('OPTIONAL ARGUMENTS')

    optional_args.add_argument('-pd', '--path-data', dest = 'path_data', required = True, type = dir_path,
                               help = 'Path to data in .mat format')
    
    optional_args.add_argument('-pcd', '--path-config_data', dest = 'path_config_data', required = True, type = file_path,
                               help = 'Path to configuration dictionnary for data.')   
    
    optional_args.add_argument('-pct', '--path-config_training', dest = 'path_config_training', required = True, type = file_path,
                               help = 'Path to configuration dictionnary for training.')
    
    optional_args.add_argument('-pm', '--path-model', dest = 'path_model', required = True, type = file_path,
                               help = 'Path to save model state.')
    
    optional_args.add_argument('-po', '--path-optimizer', dest = 'path_optimizer', required = True, type = file_path,
                               help = 'Path to save optimizer state.')
    
    optional_args.add_argument('-pc', '--path-config', dest = 'path_config', required = True, type = file_path,
                               help = 'Path to save training configuration file.')
    
    optional_args.add_argument('-gpu', '--gpu_id', dest = 'gpu_id', required = False, type = int,
                               help = 'Id of the wanted gpu device.')
    
    optional_args.add_argument('-res', '--gpu_resources', dest = 'gpu_resources', required = False, type = int,
                               help = 'Number of gpus to used per trial.')
    

    return parser


def get_tune_parser():
    
    parser = argparse.ArgumentParser(
        prog = os.path.basename(__file__).strip('.py'),
        description = 'Get command instruction and configuration file',
        add_help = True  
    )

    command_group = parser.add_mutually_exclusive_group(required = False)

    
    command_group.add_argument('--tune', dest = 'tune', action = 'store_true',
                               help = 'Tune hyperparameters on model.')

    # OPTIONAL ARGUMENTS
    optional_args = parser.add_argument_group('OPTIONAL ARGUMENTS')

    optional_args.add_argument('-pd', '--path-data', dest = 'path_data', required = True, type = dir_path,
                               help = 'Path to data in .mat format')
    
    optional_args.add_argument('-pcd', '--path-config_data', dest = 'path_config_data', required = True, type = file_path,
                               help = 'Path to configuration dictionnary for data.')   

    optional_args.add_argument('-ptc', '--path-tuning_config', dest = 'path_tuning_config', required = True, type = file_path,
                               help = 'Path to tuning configuration dictionnary for hyperparameters tuning.') 

    optional_args.add_argument('-r', '--path-results', dest = 'path_results', required = True, type = dir_path,
                               help = 'Path to folder results.')
    
    optional_args.add_argument('-pbm', '--path-best_states', dest = 'path_best_states', required = True, type = file_path,
                               help = 'Path to save best model and optimizer states.')
    
    optional_args.add_argument('-pc', '--path-best_config', dest = 'path_best_config', required = True, type = file_path,
                               help = 'Path to save best model configuration file.')
    
    optional_args.add_argument('-gpu', '--gpu_id', dest = 'gpu_id', required = False, type = int,
                               help = 'Id of the wanted gpu device.')

    optional_args.add_argument('-n', '--n_samples', dest = 'n_samples', required = False, type = int,
                               help = 'Number of samples per trial during hyperparameters search.')
    
    optional_args.add_argument('-me', '--max_n_epochs', dest = 'max_n_epochs', required = False, type = int,
                               help = 'Maximum time iteration per trial.')
    
    return parser