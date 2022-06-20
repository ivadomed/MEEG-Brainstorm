import pandas as pd
import argparse
import json
import os

from itertools import chain, combinations


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "Spike detection", description="Epileptic spike detection"
    )
    parser.add_argument("--path_root", type=str, default="../IvadomedNifti/")
    parser.add_argument("--method", type=str, nargs="+", default=["RNN_self_attention"])
    parser.add_argument("--options", type=str, nargs="+", default=[' --mix_up', ' --cost_sensitive', ' --weight_loss'])
    parser.add_argument(
        "--training", type=str, nargs="+", default=['train']
    )
    return parser


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# Experiment name
parser = get_parser()
args = parser.parse_args()
methods = args.method
trainings = args.training
options = args.options
# load data filtered
path_root = args.path_root
for training in trainings:
    for method in methods:
        for i, combo in enumerate(powerset(options), 1):
            options_combo = ''
            for option in combo:
                options_combo += option

            os.system(' python {}.py --path_root {} --save{} --method {}'.format(training, path_root, options_combo, method))