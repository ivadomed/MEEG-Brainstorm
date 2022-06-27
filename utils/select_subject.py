import os

import numpy as np
import pandas as pd

from loader.data import Data

def select_subject(n_subject, path_subject_info, path_root, len_trials):
    if not os.path.exists(path_subject_info):
        dataset = Data(path_root, 'spikeandwave', [], len_trials=len_trials)
        all_dataset = dataset.all_datasets()
        labels = all_dataset[1]
        subject_ids = np.asarray(list(labels.keys()))
        results = []
        for subject_id in subject_ids:
            label = np.concatenate(labels[subject_id])
            n_ied_segment = np.sum(label == 1)
            n_no_ied_segment = len(label) - n_ied_segment

            results.append(
                {
                    "subject_id": subject_id,
                    "type_of_segment": "no_ied",
                    "number_of_segment": n_no_ied_segment,
                }
            )
            results.append(
                {
                    "subject_id": subject_id,
                    "type_of_segment": "ied",
                    "number_of_segment": n_ied_segment,
                }
            )

        df = pd.DataFrame(results)

        if not os.path.exists(path_subject_info):
            os.mkdir(path_subject_info)

        df.to_csv(os.path.join(path_subject_info, "number_of_segment.csv"))
    
    df = pd.read_csv(os.path.join(path_subject_info, "number_of_segment.csv"))
   
    df_sort = df[df['type_of_segment']=='ied'].sort_values(by='number_of_segment')
    selected_subjects = df_sort["subject_id"].to_numpy()[-n_subject:]

    return selected_subjects