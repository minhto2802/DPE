import random
from collections import Counter

import torch

import numpy as np
import pandas as pd


def describe_dataset_splits(datasets):
    """
    Describe the distribution of 'y' (class), 'g' (subgroup), and '_a' (attribute) for each dataset split.

    Args:
        datasets (dict): Dictionary with keys as split names ('train', 'val', 'test', etc.)
                         and values as dataset objects with attributes 'y', 'g', '_a' (tensors or lists).

    Returns:
        pd.DataFrame: Multi-indexed DataFrame with counts for each category across splits.
    """
    stats = dict()

    for split, dataset in datasets.items():
        split_stats = {}
        for field in ['y', 'g', '_a']:
            values = dataset.__getattribute__(field)
            counts = Counter(values if isinstance(values, list) else values.tolist())
            for k, v in counts.items():
                split_stats[f'{field}={k}'] = v
        stats[split] = split_stats

    df = pd.DataFrame(stats).fillna(0).astype(int)
    return df.sort_index()


class DummyRun:
    def __init__(self):
        ...

    def __call__(self, *args, **kwargs):
        ...

    def log(self, *args, **kwargs):
        ...

    def finish(self, *args, **kwargs):
        ...


def get_scheduler_func(scheduler, lr, epochs, steps_per_epoch=None, pct_start=0.01):
    if scheduler == 'none':
        assert steps_per_epoch is not None

    if scheduler != 'none':
        if scheduler == 'triangle':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.CyclicLR(
                opt, 0, lr,
                step_size_up=(steps_per_epoch * epochs) // 2,
                mode='triangular', cycle_momentum=False)
        elif scheduler == 'cyclic':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.CyclicLR(
                opt, 0, lr,
                step_size_up=(steps_per_epoch * epochs) // 2,
                mode='triangular', cycle_momentum=False)
        elif scheduler == 'cosine':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * steps_per_epoch, 1e-4)
        elif scheduler == 'multistep':
            n_iters = steps_per_epoch * epochs
            milestones = [0.25 * n_iters, 0.5 * n_iters,
                          0.75 * n_iters]  # hard-coded steps for now, suitable for resnet18
            get_scheduler = lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.3)
        elif scheduler == 'onecycle':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.OneCycleLR(
                opt, lr, epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=pct_start, anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=100000,  # 2.0,
                final_div_factor=100000,  # 10000.0,
                three_phase=False,
                last_epoch=-1)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {scheduler}.")
    else:
        get_scheduler = lambda opt: None

    return get_scheduler


def fix_random_seed(seed, benchmark=False, deterministic=True):
    """Ensure reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
