from dataclasses import dataclass

import torch
import torchvision
from pandas.core.window.doc import kwargs_scipy
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from functools import partial

from . import datasets as dsets
from .eval_helpers import eval_metrics, get_acc
from .misc import get_scheduler_func, DummyRun, describe_dataset_splits, fix_random_seed
from .isomaxplus import IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart
from .datasets import Features


def get_pre_extracted_features(ckpt_dir: str, set_name: str) -> np.ndarray:
    """
    Load and normalize pre-extracted feature representations from disk.

    Parameters
    ----------
    ckpt_dir : str
        Path to the checkpoint directory where feature files are stored.
    set_name : str
        Name of the dataset split (e.g., 'train', 'val', 'test').

    Returns
    -------
    np.ndarray
        A (num_samples, feature_dim) array of L2-normalized features.
    """
    # Load memory-mapped .npy file for efficient access without full RAM loading.
    pre_extracted_feats = np.load(f'{ckpt_dir}/feats_{set_name}.npy', mmap_mode='r')

    # Normalize each feature vector to zero mean and unit variance (per sample).
    pre_extracted_feats = ((pre_extracted_feats - pre_extracted_feats.mean(axis=1, keepdims=True)) /
                           pre_extracted_feats.std(axis=1, keepdims=True))

    return pre_extracted_feats


def init_erm_model(ckpt_path, num_classes=2, device='cuda', model=None):
    """
    Initialize a pretrained ResNet50 model as an ERM backbone for feature extraction or inference.

    Parameters
    ----------
    ckpt_path : str
        Path to the saved model checkpoint (.pt or .pth file).
    num_classes : int
        Number of output classes. Default is 2 for binary classification.
    device : str
        Device identifier ('cuda' or 'cpu').
    model : Optional[torch.nn.Module]
        If provided, modifies the final classification head to match `num_classes`.

    Returns
    -------
    model : torch.nn.Module
        A sequential model with backbone, flattening, and classification head.
    """
    if model is None:
        # Load checkpoint and initialize a fresh ResNet50 base.
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = torchvision.models.resnet50()

        # Remove the final classification layer to obtain the backbone.
        backbone = torch.nn.Sequential(*list(model.children())[:-1])
        emb_dim = model.fc.in_features

        # Define a new classification head for the desired number of classes.
        head = nn.Linear(emb_dim, num_classes)

        # Compose the full model: [backbone -> flatten -> linear head]
        model = nn.Sequential(backbone, nn.Flatten(), head)

        # Load pretrained weights (including the new head, if present in ckpt).
        model.load_state_dict(ckpt, strict=True)

        # Store embedding dimension for downstream use (e.g., prototype head).
        model.emb_dim = emb_dim
    else:
        # If a model is passed, replace the head to match new num_classes.
        assert hasattr(model, "emb_dim")
        model[-1] = nn.Linear(model.emb_dim, num_classes)

    # Move to specified device for inference or training.
    model.to(device)
    return model


def evaluate(model, eval_loader, device='cuda'):
    """
    Evaluate the final classification head of a model on pre-extracted features.

    Parameters
    ----------
    model : torch.nn.Module
        A model with the last layer being a classifier (typically the prototype or linear head).
    eval_loader : DataLoader
        A PyTorch DataLoader over pre-extracted feature tensors (not raw images).
    device : str
        Device to run inference on. Defaults to 'cuda'.

    Returns
    -------
    dict
        A dictionary of performance metrics including accuracy per group, class, and attribute,
        as computed by `eval_metrics()`.
    """
    ds = eval_loader.dataset
    classes, attributes, groups = np.array(ds.y), np.array(ds._a), np.array(ds.g)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for *_, feats in eval_loader:
            feats = feats.to(device)
            outputs = model(feats)  # Assume model is [backbone, ..., head]; use head only
            all_preds.append(outputs.detach().softmax(1).cpu())

        all_preds = torch.concat(all_preds, dim=0).numpy()

        # Compute per-group/class metrics
        res = eval_metrics(all_preds, np.array(classes), np.array(attributes), np.array(groups))

    return res


def get_subsampled_train_set(
        datasets=None,
        trn_split='va',
        *args, **kwargs,
):
    """
    Initialize a subgroup-balanced or attribute-balanced training dataset.

    Parameters
    ----------
    datasets : dict or None
        A dictionary of pre-loaded datasets. If None, a new dict is created.
    data_dir : str
        Root directory of all datasets.
    train_attr : str
        One of {'yes', 'no'}. Determines whether attribute information is used in subsampling.
    subsample_type : str
        Strategy for subsampling (e.g., 'group', 'none').
    dataset_name : str
        Dataset class name from `utils.datasets` (e.g., 'Waterbirds', 'CelebA').
    trn_split : str
        Data split to use for training (e.g., 'va' for validation-as-training).

    Returns
    -------
    dict
        Updated `datasets` dictionary including the subsampled 'train' split.
    """

    if datasets is None:
        pre_extracted_feats = None
        datasets = {}
    else:
        pre_extracted_feats = datasets['val'].feats

    datasets['train'] = vars(dsets)[kwargs['dataset_name']](
        split=trn_split,
        pre_extracted_feats=pre_extracted_feats,
        *args, **kwargs,
    )
    return datasets


def get_train_loader(datasets=None, train_attr='yes', batch_size=256, workers=8, dataset_name='Waterbirds',
                     *args, **kwargs) -> DataLoader:
    """
    Construct a PyTorch DataLoader for training with subsampled training data.

    Parameters
    ----------
    datasets : dict
        Dictionary of dataset splits (expects at least 'val', and will create 'train').
    train_attr : str
        Whether to use attribute annotations for balancing ('yes' or 'no').
    batch_size : int
        Batch size for training.
    workers : int
        Number of parallel DataLoader workers.
    dataset_name : str
        Name of the dataset class (e.g., 'Waterbirds').

    Returns
    -------
    DataLoader
        A PyTorch DataLoader for the 'train' subset.
    """
    datasets = get_subsampled_train_set(
        datasets,
        train_attr=train_attr,
        dataset_name=dataset_name,
        *args, **kwargs,
    )
    describe_dataset_splits(datasets)

    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=workers,
        pin_memory=False
    )

    return train_loader


def init_model(ckpt_path=None, num_classes=2, model=None, device='cuda'):
    """
    Initialize or modify a ResNet-50 model with IsoMax+ classification head.

    Parameters
    ----------
    ckpt_path : str
        Path to the checkpoint file containing pretrained weights.
    num_classes : int
        Number of output classes.
    model : torch.nn.Module or None
        If None, a new model is initialized; otherwise, final layer is replaced.
    device : str
        Device to move model to.

    Returns
    -------
    torch.nn.Module
        A ResNet-50 backbone with a distance-based IsoMaxPlusLossFirstPart head.
    """
    if model is None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = torchvision.models.resnet50()
        backbone = torch.nn.Sequential(*list(model.children())[:-1])
        emb_dim = model.fc.in_features
        head = IsoMaxPlusLossFirstPart(emb_dim, num_classes)

        model = nn.Sequential(backbone, nn.Flatten(), head)
        model.load_state_dict(ckpt, strict=False)
        model.emb_dim = emb_dim
    else:
        assert hasattr(model, "emb_dim")
        model[-1] = IsoMaxPlusLossFirstPart(model.emb_dim, num_classes)
    model.to(device)
    return model


def train_prototypes(train_loader, val_loader, model, prototype_ensemble=(),
                     epochs=20, cov_reg=5e5, wd_weight=10, device='cuda',
                     entropic=30, lr=1e-3, stage=1, verbose=True, loss_name='isomax', optim='sgd',
                     scheduler='none', *args, **kwargs):
    """
    Train a single prototype head using supervised data and optionally diversify from previous heads.

    Parameters
    ----------
    train_loader : DataLoader
        Loader for training samples (features only).
    val_loader : DataLoader
        Loader for validation data (used to select best prototype).
    model : torch.nn.Module
        Backbone + prototype classifier head.
    prototype_ensemble : list of tuple
        List of previous prototype tensors and distance scales.
    epochs : int
        Number of training epochs.
    cov_reg : float
        Weight on the covariance regularization term for inter-prototype diversity.
    wd_weight : float
        Weight decay multiplier for L2 penalty on prototypes.
    device : str
        CUDA or CPU device identifier.
    entropic : float
        Entropic scale parameter for IsoMax loss.
    lr : float
        Learning rate for SGD.
    stage : int
        Current stage in ensemble construction.
    verbose : bool
        Whether to show tqdm and metrics.
    loss_name : str
        Either 'isomax' or 'ce'.
    optim : str
        Either 'adam' or 'sgd'.
    scheduler : str
        ['none', 'onecycle']

    Returns
    -------
    tuple
        Best prototype head: (prototypes, distance_scale) for IsoMax or (weights, bias) for CE.
    """
    best_val_wga, val_wga = 0.0, 0.0
    best_val_wga_prototype = None

    if loss_name == 'isomax':
        criterion = IsoMaxPlusLossSecondPart(entropic_scale=entropic, reduction='mean')
    else:
        criterion = CrossEntropyLoss(reduction='mean')
    match optim:
        case 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        case 'adam':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    scheduler = get_scheduler_func(
        scheduler, lr, epochs, len(train_loader))(optimizer)

    if len(prototype_ensemble) > 0:
        prototype_ensemble = torch.concat([_[0] for _ in prototype_ensemble], dim=1).detach()

    pbar = tqdm(range(epochs), desc=f'[Stage {stage}]')
    for epoch in pbar:
        model.train()
        running_loss, running_clf, running_cov, running_correct, total = 0.0, 0.0, 0.0, 0, 0

        for _, _, labels, _, feats in train_loader:
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(feats)
            clf_loss = criterion(outputs, labels)

            head = model[-1]
            cov_loss = torch.tensor(0.0, device=device)

            if isinstance(criterion, IsoMaxPlusLossSecondPart):
                wd = torch.einsum('ijk,ilk->ijl', [head.prototypes[:, None], head.prototypes[:, None]]) * wd_weight
                wd = wd.squeeze().mean()
                loss = clf_loss + wd

                if len(prototype_ensemble) and (cov_reg > 0):
                    _prototypes = torch.cat([head.prototypes[:, None], prototype_ensemble], dim=1)
                    n_pro, n_dim = _prototypes.shape[1:]
                    cov = torch.einsum('ijk,ilk->ijl', [_prototypes, _prototypes]) / (n_dim - 1)
                    cov_loss = torch.abs(cov[:, 0, 1:].sum(1).div(n_pro).mean())
                    loss += cov_loss * cov_reg
            else:
                weight = head.weight  # if loss_name == 'ce'
                loss = clf_loss + 0.1 * torch.norm(weight, 1)

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            preds = outputs.argmax(dim=1)

            correct = (preds == labels).sum().item()
            running_loss += loss.item()
            running_clf += clf_loss.item()
            running_cov += cov_loss.item()
            running_correct += correct
            total += labels.size(0)

        val_wga = evaluate(model, val_loader, device=device)['min_group']['accuracy']
        if val_wga >= best_val_wga:
            best_val_wga = val_wga
            if loss_name == 'isomax':
                best_val_wga_prototype = [
                    model[-1].prototypes[:, None].detach().clone(),
                    model[-1].distance_scale.detach().clone()
                ]
            else:
                best_val_wga_prototype = [
                    model[-1].weight[:, None].detach().clone(),
                    model[-1].bias.detach().clone(),
                ]

    return best_val_wga_prototype


def cov_reg_scheduler_inv(cov_reg, num_stages, alpha=0.05):
    """
    Construct a decreasing schedule for the covariance regularization coefficient.

    Parameters
    ----------
    cov_reg : float
        Initial regularization strength.
    num_stages : int
        Number of ensemble stages.
    alpha : float
        Decay rate for inverse schedule.

    Returns
    -------
    function
        A callable that maps `stage` to regularization weight.
    """
    scheduler = [cov_reg / (1 + alpha * _) for _ in range(num_stages)]
    scheduler = [cov_reg] + scheduler

    def cov_reg_schedule(stage):
        stage = max(stage, 1)
        return scheduler[stage - 1]

    return cov_reg_schedule


def train_ensemble(
        init_model_func,
        datasets,
        dataloaders,
        init_train_loader,
        cov_reg=5e5,
        random_subset=True,
        num_stages=15,
        show_freq=15,
        full_model=None,
        optim='sgd',
        alpha=0.1,
        run=None,
        *args, **kwargs
):
    """
    Full ensemble training loop for Diversified Prototypical Ensemble (DPE).

    Parameters
    ----------
    init_model_func : Callable
        Function to initialize or modify model.
    datasets : dict
        Dataset splits for train/val/test.
    dataloaders : dict
        Dataloaders for validation and test.
    init_train_loader : Callable
        Function to generate train DataLoader.
    cov_reg : float
        Initial covariance regularization weight.
    random_subset : bool
        Whether to randomly resample training data at each stage.
    num_stages : int
        Number of prototype ensemble members to train.
    show_freq : int
        How often to display training progress.
    alpha : covariance decay
    run : covariance scheduler
    full_model : nn.Module, could be nn.Sequential(nn.Identity(), nn.Identity()) if no backbone is not required,
                or nn.Sequential(backbone, nn.Identity()) otherwise.
    optim: 'sgd' or 'adam'.

    Returns
    -------
    tuple
        Final worst-group, balanced, and detailed evaluation results.
    """
    run = DummyRun() if run is None else run
    prototype_ensemble = []
    ensemble_wga, ensemble_acc, ensemble_balanced_acc = [], [], []
    _train_prototypes = partial(
        train_prototypes,
        val_loader=dataloaders['val'],
        optim=optim,
        *args, **kwargs
    )
    cv_scheduler = cov_reg_scheduler_inv(cov_reg, num_stages, alpha=alpha)
    train_loader = init_train_loader(datasets)

    for stage in range(1, num_stages + 1):
        verbose = stage % show_freq == 0 or stage == 1 or stage == num_stages
        if verbose:
            print(describe_dataset_splits({k: dataloaders[k].dataset for k in dataloaders.keys()}))

        full_model = init_model_func(model=full_model)
        prototype_ensemble.append(_train_prototypes(
            train_loader, model=full_model,
            prototype_ensemble=prototype_ensemble, stage=stage,
            cov_reg=cv_scheduler(stage),
        ))
        res = evaluate_ensemble(
            prototype_ensemble,
            dataloaders['test'],
            full_model,
            device=kwargs['device'],
            verbose=verbose
        )
        ensemble_wga.append(res['min_group']['accuracy'])
        ensemble_acc.append(res['overall']['accuracy'])
        ensemble_balanced_acc.append(res['overall']['balanced_acc'])

        run.log({
            "dpe_test/stage": stage,
            "dpe_test/cv_scheduler": cv_scheduler(stage),
            "dpe_test/worst_group_accuracy": ensemble_wga[-1] * 100,
            "dpe_test/accuracy": ensemble_acc[-1] * 100,
            "dpe_test/balanced_accuracy": ensemble_balanced_acc[-1] * 100
        })

        if stage <= num_stages + 1 and random_subset:
            train_loader = init_train_loader(datasets)

    return ensemble_wga, ensemble_balanced_acc, res, prototype_ensemble


def evaluate_ensemble(prototype_ensemble, eval_loader, model,
                      device='cuda', show_individuals=False, verbose=True):
    """
    Evaluate the full prototype ensemble by averaging predictions over all members.

    Parameters
    ----------
    prototype_ensemble : list
        List of (prototypes, distance_scale) or (weights, bias) tuples.
    eval_loader : DataLoader
        Test DataLoader over pre-extracted features.
    model : torch.nn.Module
        The model to overwrite classifier parameters for inference.
    device : str
        Evaluation device.
    show_individuals : bool
        If True, print accuracy per member.
    verbose : bool
        If True, display final WGA score.

    Returns
    -------
    dict
        Evaluation metrics including worst-group accuracy.
    """
    dist_scales = [_[1].detach() for _ in prototype_ensemble]
    clf = torch.concat([_[0] for _ in prototype_ensemble], dim=1).detach().transpose(0, 1)
    preds_list = torch.zeros(clf.shape[0], len(eval_loader.dataset), eval_loader.dataset.num_labels)

    ds = eval_loader.dataset
    classes, attributes, groups = np.array(ds.y), np.array(ds._a), np.array(ds.g)

    position = 0

    with torch.no_grad():
        for *_, feats in tqdm(eval_loader, leave=False):
            feats = feats.to(device)

            for i, weight in enumerate(clf):
                if hasattr(model[-1], 'prototypes'):
                    model[-1].prototypes = torch.nn.Parameter(weight, requires_grad=False)
                    model[-1].distance_scale = nn.Parameter(dist_scales[i], requires_grad=False)
                else:
                    model[-1].weight = torch.nn.Parameter(weight, requires_grad=False)
                    model[-1].bias = torch.nn.Parameter(dist_scales[i], requires_grad=False)  # bias for ce loss
                model.eval()
                preds_list[i][position:position + feats.shape[0]] = model(feats.squeeze())
            position += feats.shape[0]

    if show_individuals:
        for i in range(preds_list.shape[0] - 1, -1, -1):
            preds = preds_list[i].softmax(1).argmax(1).numpy()
            get_acc(preds, classes, groups)

    get_acc(preds_list.softmax(2).mean(0).argmax(1).numpy(), classes, groups, verbose=verbose)

    preds = preds_list.softmax(2).mean(0).detach().cpu().numpy()
    res = eval_metrics(preds, classes, attributes, groups)
    if verbose:
        print(f"Ensemble WGA: {res['min_group']['accuracy'] * 100:.1f}")
    return res


@dataclass(frozen=True)
class Args:
    data_dir: str = ""
    metadata_path: str = ""
    num_classes: int = None
    norm_emb: bool = True
    dataset_name: str = 'Features'
    device: str = 'cuda'
    workers: int = 0
    batch_size_train: int = 256
    batch_size_eval: int = 256
    train_attr: str = 'no'
    seed: int = 42
    epochs: int = 20
    lr: float = 1e-3
    multi_class: bool = False  # requires column 'yy' in the metadata
    num_stages: int = 15
    emb_dim: int = 2048
    split_map: dict = None
    scheduler: str = 'none'
    cov_reg: float = 5e4
    entropic: int = 30
    show_freq: int = 10
    optim: str = 'sgd'
    trn_split: str = 'va'
    loss_name: str = 'isomax'  # ce
    subsample_type: str = 'group'
    verbose: bool = False
    alpha: float = 0.05
    # d_model: int = 256
    # ff_dim: int = 1024


class DPE:
    def __init__(self, *args, **kwargs):
        self.config = Args(*args, **kwargs)
        self.datasets, self.loaders = dict(), dict()
        self.set_loaders()
        self.ensemble = None

    def set_loaders(self, datasets=None):
        if datasets is None:
            self._init_dataset(**vars(self.config))
            datasets = self.datasets
        else:
            self.datasets = datasets

        for set_name in datasets:
            if set_name == 'train':
                continue
            self.loaders[set_name] = DataLoader(
                dataset=datasets[set_name],
                num_workers=self.config.workers,
                pin_memory=False,
                batch_size=self.config.batch_size_eval,
                shuffle=False,
                drop_last=False
            )
        if self.config.verbose:
            describe_dataset_splits(datasets)

    def _init_dataset(
            self,
            data_dir=None,
            metadata_path=None,
            split_map=None,
            norm_emb=True,
            transform=None,
            dataset_name='Features',
            *args, **kwargs,
    ):

        assert data_dir is not None and metadata_path is not None

        split_map = {'val': 'va', 'test': 'te'} if split_map is None else split_map
        kwargs['subsample_type'] = None
        for split in split_map.keys():
            features = np.load(f"{data_dir}/feats_{split}.npy")
            if norm_emb:
                features = ((features - features.mean(axis=1, keepdims=True)) / features.std(axis=1, keepdims=True))
            self.datasets[split] = vars(dsets)[dataset_name](
                data_dir=data_dir,
                metadata_path=metadata_path,
                split=split_map[split],
                transform=transform,
                pre_extracted_feats=features,
                *args, **kwargs,
            )

    def fit(self):
        clf_head = nn.Sequential(nn.Identity(), nn.Identity())
        clf_head.emb_dim = self.config.emb_dim
        clf_head.to(self.config.device)

        *metrics, self.ensemble = train_ensemble(
            datasets=self.datasets,
            dataloaders=self.loaders,
            init_train_loader=partial(
                get_train_loader,
                batch_size=self.config.batch_size_train,
                **vars(self.config),
            ),
            full_model=clf_head,
            init_model_func=partial(
                init_model,
                device=self.config.device,
                num_classes=self.datasets['val'].num_labels,
            ),
            **vars(self.config),
        )
        return metrics
