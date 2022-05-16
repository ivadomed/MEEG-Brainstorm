import pandas as pd
import argparse
import json
import os


def get_parser():
    """Set parameters for the experiment."""
    parser = argparse.ArgumentParser(
        "Spike detection", description="Domain Adaptation on sleep EEG."
    )
    parser.add_argument("--path_data", type=str, default="../IvadomedNifti/")

    return parser


# Experiment name
parser = get_parser()
args = parser.parse_args()

# load data filtered
path_data = args.path_data

with open(os.path.join(path_data, 'config_for_training.json'), 'r') as f:
    config_json = json.load(f)
path_output = config_json["path_output"]
config_json["split_dataset"]["data_testing"]["data_type"] = "participant_id"

df = pd.read_csv(os.path.join(path_data, "participants.tsv"), sep='\t')

participant_ids = df['participant_id'].to_numpy()

for participant_id in participant_ids:

    config_json["split_dataset"]["data_testing"]["data_value"] = participant_id
    config_json["path_output"] = path_output + '_' + participant_id
    config_json_name = f"config_for_training_{participant_id}.json"

    with open(config_json_name, 'w') as json_file:
        json.dump(config_json, json_file, indent=4, sort_keys=True)

    os.system(f'ivadomed -c {config_json_name}')
