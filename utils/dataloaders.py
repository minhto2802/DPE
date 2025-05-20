import torch


import numpy as np
from functools import partial
from torch.utils.data import DataLoader


def make_weights_for_balanced_classes(target):
    class_sample_count = torch.tensor([(target == t).sum() for t in torch.unique(target, sorted=True)])
    weight = 1. / class_sample_count.float()
    try:
        return torch.tensor([weight[t] for t in target])
    except IndexError as E:
        return torch.zeros_like(target)


def get_balanced_batch_sampler(dataset):
    try:
        targets = dataset.y if dataset.subsample_target == 'y' else dataset.g
    except:
        targets = np.array(dataset.y)[dataset.idx]
    weights = make_weights_for_balanced_classes(torch.tensor(targets))
    batch_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return batch_sampler


def get_dataloaders(datasets, batch_size_train=128, batch_size_eval=256, workers=0, stage=0,
                    no_class_balanced_training=False, shuffle_in_training=False):
    dataloaders = {}

    for set_name, dataset in datasets.items():
        if set_name == 'train':
            if stage == 0:
                sampler = get_balanced_batch_sampler(dataset)
                shuffle = None
            else:
                if no_class_balanced_training:
                    sampler = None
                    shuffle = shuffle_in_training
                else:
                    sampler = get_balanced_batch_sampler(dataset)
                    shuffle = False
            dl = partial(DataLoader, batch_size=batch_size_train, sampler=sampler,
                         drop_last=True, shuffle=shuffle)
        else:
            dl = partial(DataLoader, batch_size=batch_size_eval, shuffle=False)
        dataloaders[set_name] = dl(dataset=dataset, num_workers=workers, pin_memory=False)
    return dataloaders
