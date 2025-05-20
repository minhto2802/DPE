import time
import random
import functools

import torch
import numpy as np
import pandas as pd
from torch import nn

from .optimizers import bert_lr_scheduler
from .isomaxplus import IsoMaxPlusLossSecondPart


def fix_random_seed(seed, deterministic=True, benchmark=True):
    """

    :param seed:
    :param deterministic:
    :param benchmark:
    :return:
    """
    import torch, random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def off_diagonal(x):
    """

    :param x:
    :return:
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def none_or_str(value):
    if value.lower() in ['none', 'null']:
        return None
    return value


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.4f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]

    print(sep.join([format_val(x) for x in row]), end_)


class DummyRun:
    def __init__(self):
        ...

    def __call__(self, *args, **kwargs):
        ...

    def log(self, *args, **kwargs):
        ...


def get_scheduler_func(scheduler, lr, epochs, steps_per_epoch=None):
    if scheduler == 'none':
        assert steps_per_epoch is not None

    if scheduler != 'none':
        if scheduler == 'triangle':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.CyclicLR(
                opt, 0, lr,
                step_size_up=(steps_per_epoch * epochs) // 2,
                mode='triangular', cycle_momentum=False)
        elif scheduler == 'cosine':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.CyclicLR(
                opt, 0, lr,
                step_size_up=(steps_per_epoch * epochs) // 2,
                mode='cosine', cycle_momentum=False)
        elif scheduler == 'multistep':
            n_iters = steps_per_epoch * epochs
            milestones = [0.25 * n_iters, 0.5 * n_iters,
                          0.75 * n_iters]  # hard-coded steps for now, suitable for resnet18
            get_scheduler = lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.3)
        elif scheduler == 'onecycle':
            get_scheduler = lambda opt: torch.optim.lr_scheduler.OneCycleLR(
                opt, lr, epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.01, anneal_strategy='cos',
                cycle_momentum=True,
                base_momentum=0.85,
                max_momentum=0.95,
                div_factor=100000,  # 2.0,
                final_div_factor=100000,  # 10000.0,
                three_phase=False,
                last_epoch=-1)
        elif scheduler == 'bert':
            get_scheduler = lambda opt: bert_lr_scheduler(opt, epochs)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {scheduler}.")
    else:
        get_scheduler = lambda opt: None

    return get_scheduler


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def get_criterion(loss_name, reduction='mean', entropic_scale=20):
    if loss_name == 'ce':
        criterion = nn.CrossEntropyLoss(reduction=reduction)
    elif loss_name == 'isomax':
        criterion = IsoMaxPlusLossSecondPart(entropic_scale=entropic_scale, reduction=reduction)
    else:
        raise NotImplementedError()
    return criterion
