import numpy as np

from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import LeavePGroupsOut



def pick_recordings(dataset, subj_rec_nbs):
    """Pick recordings using subject and recording numbers.
    Parameters
    ----------
    dataset : ConcatDataset
        The dataset to pick recordings from.
    subj_rec_nbs : list of tuples
        List of pairs (subj_nb, rec_nb) to use in split.
    Returns
    -------
    ConcatDataset
        The picked recordings.
    ConcatDataset | None
        The remaining recordings. None if all recordings from
        `dataset` were picked.
    """
    pick_idx = list()
    for subj_nb, rec_nb in subj_rec_nbs:
        for i, ds in enumerate(dataset.datasets):
            if (ds.subj_nb == subj_nb) and (ds.rec_nb == rec_nb):
                pick_idx.append(i)

    pick_ds = ConcatDataset([dataset.datasets[i] for i in pick_idx])

    return pick_ds


def train_test_split(dataset, n_groups, split_by="subj_nb"):
    """Split dataset into train and test keeping n_groups out in test.
    Parameters
    ----------
    dataset : ConcatDataset
        The dataset to split.
    n_groups : int
        The number of groups to leave out.
    split_by : 'subj_nb' | 'rec_nb'
        Property to use to split dataset.
    Returns
    -------
    ConcatDataset
        The training data.
    ConcatDataset
        The testing data.
    """
    groups = [getattr(ds, split_by) for ds in dataset.datasets]
    train_idx, test_idx = next(LeavePGroupsOut(n_groups).split(X=groups, groups=groups))
    train_ds = ConcatDataset([dataset.datasets[i] for i in train_idx])
    test_ds = ConcatDataset([dataset.datasets[i] for i in test_idx])

    return train_ds, test_ds


def loaders_LOPO(
    model,
    dataset_source,
    dataset_target,
    rec_subjects,
    subjects_source,
    subject_target,
    batch_size,
    num_workers,
    mix_source=True,
    nb_val=1,
):

    source_ds = list()
    for k in subjects_source:
        source_ds.append(pick_recordings(dataset_source, [(k, 1), (k, 2)]))

    if rec_subjects[subject_target] == 1:
        # Take the only rec available
        target_dss = [
            pick_recordings(dataset_target, [(subject_target, 1), (subject_target, 2)])
        ]
    else:
        target_dss = [
            pick_recordings(dataset_target, [(subject_target, 1)]),
            pick_recordings(dataset_target, [(subject_target, 2)]),
        ]
    # Find the closest subjects of target sucject

    valid_ds = pick_recordings(
        dataset_source,
        [
            (subjects_source[closest_patient[i]], j)
            for i in range(nb_val)
            for j in range(1, 3)
        ],
    )

    subjects_train = np.delete(
        subjects_source, [closest_patient[i] for i in range(nb_val)]
    )
    
    source_dss = [pick_recordings(
        dataset_source, [(i, j) for i in subjects_train for j in range(1, 3)]
    )]
    # Define class weight
    train_y = np.concatenate([ds.epochs_labels for ds in source_dss[0].datasets])
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_y), y=train_y
    )  

    if not mix_source:
        source_dss = list()
        for k in subjects_train:
            source_dss.append(pick_recordings(dataset_source, [(k, 1), (k, 2)]))
            
    # Define loaders
    loader_valid = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )

    loaders_source = [
        DataLoader(
            source_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        for source_ds in source_dss
    ]
    loaders_target = [
        DataLoader(
            target_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        )
        for target_ds in target_dss
    ]

    return loaders_source, loaders_target, loader_valid, class_weights


def loaders_DA(dataset_s, dataset_t, subject_s, subject_t, batch_size, num_workers):

    # Loaders source (train, valid)
    n_train = int(0.80 * len(subject_s))

    source_ds = pick_recordings(
        dataset_s, [(s, r) for s in subject_s[:n_train] for r in [1, 2]]
    )
    n_subjects_valid = max(1, int(len(source_ds.datasets) * 0.2))
    source_ds, valid_ds = train_test_split(
        source_ds, n_subjects_valid, split_by="subj_nb"
    )

    train_y = np.concatenate([ds.epochs_labels for ds in source_ds.datasets])
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_y), y=train_y
    )

    loader_source = DataLoader(
        source_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    loader_valid = DataLoader(
        valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # loaders target
    n_train = int(0.80 * len(subject_t))

    target_ds = pick_recordings(
        dataset_t, [(s, r) for s in subject_t[n_train:] for r in [1, 2]]
    )

    loader_target = DataLoader(
        target_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Define class weight
    train_y = np.concatenate([ds.epochs_labels for ds in source_ds.datasets])
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_y), y=train_y
    )

    return loader_source, loader_target, loader_valid, class_weights