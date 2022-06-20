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
        raise argparse.ArgumentTypeError('readable_dir: {} '
                                         'is not a valid '
                                         'direction path'.format(path))


def file_path(path):
    return path


def get_parser():

    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__).strip('.py'),
        description='Get command instruction and configuration file',
        add_help=True
    )

    # COMMAND ARGUMENTS

    command_group = parser.add_mutually_exclusive_group(required=True)

    command_group.add_argument('--train', dest='train', action='store_true',
                               help='Perform cross-validation on data.')

    command_group.add_argument('--test', dest='test', action='store_true',
                               help='Perform testing on trained model.')

    # ARGUMENTS

    parser.add_argument('--path-root', dest='path_root', required=True,
                        type=dir_path, help='Path to data in .mat format')

    parser.add_argument('--path-config', dest='path_config', required=True,
                        type=file_path,
                        help='Path to configuration file in .json format')

    parser.add_argument('--path-output', dest='path_output', required=True,
                        type=dir_path, help='Path to output folder.')

    parser.add_argument('--gpu_id', dest='gpu_id', required=True, type=int,
                        help='Id of the cuda device to use if available.')

    # OPTIONAL ARGUMENTS

    optional_args = parser.add_argument_group('OPTIONAL ARGUMENTS')

    optional_args.add_argument('--save', dest='save', action='store_true',
                               required=False,
                               help='Save training and validation information,'
                               ' model parameters and config file.')

    return parser
