from functools import partial
from collections import Counter
from dataclasses import dataclass, field, fields

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import datasets as dsets
from .eval_helpers import eval_metrics
from .misc import get_scheduler_func, fix_random_seed
from .isomaxplus import IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart


class DPE:
    """
    Diversified Prototypical Ensemble (DPE).

    Trains an ensemble of prototype classifiers to improve robustness under subpopulation shifts.
    """
    _meta = {
        '__name__': 'DPE',
        '__version__': '0.2.0',
        '__author__': 'Minh To',
        '__license__': 'MIT',
        '__url__': 'https://github.com/minhto2802/dpe4subpop',
        '__requirements__': [
            'torch>=1.9.0',
            'torchvision>=0.9.0',
            'numpy>=1.19.0',
            'pandas>=1.1.0',
            'tqdm>=4.50.0',
        ],
    }

    def __init__(self, *args, **kwargs):
        self.config = Args(*args, **kwargs)
        if self.config.seed is not None:
            fix_random_seed(self.config.seed)

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

    def describe_dataset_splits(self):
        """
        Describe the distribution of 'y' (class), 'g' (subgroup), and '_a' (attribute) for each dataset split.

        Args:
            datasets (dict): Dictionary with keys as split names ('train', 'val', 'test', etc.)
                             and values as dataset objects with attributes 'y', 'g', '_a' (tensors or lists).

        Returns:
            pd.DataFrame: Multi-indexed DataFrame with counts for each category across splits.
        """
        stats = dict()

        for split, dataset in self.datasets.items():
            split_stats = {}
            for field in ['y', 'g', '_a']:
                values = dataset.__getattribute__(field)
                counts = Counter(values if isinstance(values, list) else values.tolist())
                for k, v in counts.items():
                    split_stats[f'{field}={k}'] = v
            stats[split] = split_stats

        df = pd.DataFrame(stats).fillna(0).astype(int)
        return df.sort_index()

    def _init_shelf_model(self):
        """
        Initialize a shelf model with an identity head for the IsoMax+ loss.
        This is used to train the ensemble of prototypes.
        """
        clf_head = nn.Sequential(nn.Identity(), nn.Identity())
        clf_head.emb_dim = self.config.emb_dim
        clf_head.to(self.config.device)
        return clf_head

    def fit(self) -> list:
        clf_head = self._init_shelf_model()

        *metrics, self.ensemble = _train_ensemble(
            datasets=self.datasets,
            dataloaders=self.loaders,
            init_train_loader=partial(
                _get_train_loader,
                batch_size=self.config.batch_size_train,
                **vars(self.config),
            ),
            full_model=clf_head,
            init_model_func=partial(
                _init_model,
                device=self.config.device,
                num_classes=self.datasets['val'].num_labels,
                loss_name=self.config.loss_name,
            ),
            **vars(self.config),
        )
        return metrics

    def evaluate(self, target='test') -> dict:
        """
        Predict using the trained ensemble on the test set.

        Returns
        -------
        dict
            A dictionary of performance metrics including accuracy per group, class, and attribute,
            as computed by `eval_metrics()`.
        """
        assert target in self.loaders, f"Target '{target}' not found in loaders."

        ds = self.loaders[target].dataset
        classes, attributes, groups = np.array(ds.y), np.array(ds._a), np.array(ds.g)

        preds = self.predict_proba(target=target).cpu().numpy()
        res = eval_metrics(preds, classes, attributes, groups)
        return res

    def predict_proba(self, target='test', avg=True, raw_logits=False) -> torch.Tensor:
        """
        Predict probabilities using the trained ensemble on the test set.

        Returns
        -------
        torch.Tensor
            An array of predicted probabilities for each class.
        """
        assert self.ensemble is not None, "Model must be trained before predicting."

        model = _init_model(
            num_classes=self.datasets['val'].num_labels,
            model=self._init_shelf_model(),
            loss_name=self.config.loss_name,
            device=self.config.device,
        )

        pred_list = _predict(
            prototype_ensemble=self.ensemble,
            eval_loader=self.loaders[target],
            model=model,
            device=self.config.device,
        )

        if not raw_logits:
            pred_list = pred_list.softmax(dim=2)

        if avg:
            return pred_list.mean(0).detach()
        return pred_list

    @staticmethod
    def help():
        """List all arguments with their descriptions and default values."""
        Args.help()


def _evaluate(model, eval_loader, device='cuda'):
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


def _get_subsampled_train_set(
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


def _get_train_loader(datasets=None, train_attr=False, batch_size=256, workers=8, dataset_name='Waterbirds',
                      *args, **kwargs) -> DataLoader:
    """
    Construct a PyTorch DataLoader for training with subsampled training data.

    Parameters
    ----------
    datasets : dict
        Dictionary of dataset splits (expects at least 'val', and will create 'train').
    train_attr : bool
        Whether to use attribute annotations for balancing.
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
    datasets = _get_subsampled_train_set(
        datasets,
        train_attr=train_attr,
        dataset_name=dataset_name,
        *args, **kwargs,
    )

    train_loader = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=workers,
        pin_memory=False
    )

    return train_loader


def _init_model(num_classes=2, model=None, loss_name='isomax', device='cuda'):
    """
    Initialize IsoMax+ classification head.

    Parameters
    ----------
    num_classes : int
        Number of output classes.
    device : str
        Device to move model to.

    Returns
    -------
    torch.nn.Module
        A ResNet-50 backbone with a distance-based IsoMaxPlusLossFirstPart head.
    """
    assert hasattr(model, "emb_dim")
    if loss_name == 'isomax':
        model[-1] = IsoMaxPlusLossFirstPart(model.emb_dim, num_classes)
    elif loss_name == 'ce':
        model[-1] = nn.Linear(model.emb_dim, num_classes)
    else:
        raise ValueError(f"Unsupported loss name: {loss_name}")
    model.to(device)
    return model


def _train_prototypes(train_loader, val_loader, model, prototype_ensemble=(),
                      epochs=20, cov_reg=5e5, wd_weight=10, device='cuda',
                      entropic=30, lr=1e-3, loss_name='isomax', optim='sgd',
                      scheduler='none', weight_decay=0.0, *args, **kwargs):
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
    loss_name : str
        Either 'isomax' or 'ce'.
    optim : str
        Either 'adam' or 'sgd'.
    scheduler : str
        ['none', 'onecycle']
    weight_decay : float
        Weight decay for the optimizer.

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
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        case 'adam':
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = get_scheduler_func(
        scheduler, lr, epochs, len(train_loader))(optimizer)

    if len(prototype_ensemble) > 0:
        prototype_ensemble = torch.concat([_[0] for _ in prototype_ensemble], dim=1).detach()

    for epoch in range(epochs):
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

        val_wga = _evaluate(model, val_loader, device=device)['min_group']['accuracy']
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


def _train_ensemble(
        init_model_func,
        datasets,
        dataloaders,
        init_train_loader,
        cov_reg=5e5,
        random_subset=True,
        num_stages=15,
        full_model=None,
        optim='sgd',
        alpha=0.1,
        eval_freq=-1,
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
    eval_freq : int
        Frequency of online evaluation of the ensemble during training. -1 means no online eval.
    alpha : covariance decay
    full_model : nn.Module, could be nn.Sequential(nn.Identity(), nn.Identity()) if no backbone is not required,
                or nn.Sequential(backbone, nn.Identity()) otherwise.
    optim: 'sgd' or 'adam'.

    Returns
    -------
    tuple
        Final worst-group, balanced, and detailed evaluation results.
    """
    prototype_ensemble = []
    res = None
    ensemble_wga, ensemble_acc, ensemble_balanced_acc = [], [], []

    train_prototypes = partial(
        _train_prototypes,
        val_loader=dataloaders['val'],
        optim=optim,
        *args, **kwargs
    )
    cv_scheduler = cov_reg_scheduler_inv(cov_reg, num_stages, alpha=alpha)
    train_loader = init_train_loader(datasets)

    pbar = tqdm(range(1, num_stages + 1), desc=f'[Training]')

    for stage in pbar:
        pbar.set_description(f'[Stage {stage}]')

        full_model = init_model_func(model=full_model)
        prototype_ensemble.append(train_prototypes(
            train_loader, model=full_model,
            prototype_ensemble=prototype_ensemble,
            cov_reg=cv_scheduler(stage),
        ))
        if (eval_freq > 0) and (stage % eval_freq == 0 or stage == num_stages):
            res = _evaluate_ensemble(prototype_ensemble, dataloaders['test'], full_model, device=kwargs['device'])

            ensemble_wga.append(res['min_group']['accuracy'])
            ensemble_acc.append(res['overall']['accuracy'])
            ensemble_balanced_acc.append(res['overall']['balanced_acc'])

            pbar.set_postfix({
                'wga': f"{ensemble_wga[-1] * 100:.1f}",
                'acc': f"{ensemble_acc[-1] * 100:.1f}",
                'bacc': f"{ensemble_balanced_acc[-1] * 100:.1f}",
            })

        if stage <= num_stages + 1 and random_subset:
            train_loader = init_train_loader(datasets)

    return ensemble_wga, ensemble_balanced_acc, res, prototype_ensemble


def _predict(prototype_ensemble, eval_loader, model, device='cuda'):
    """
    Predict using a trained prototype ensemble on pre-extracted features.
    Parameters
    ----------
    prototype_ensemble : list
        List of (prototypes, distance_scale) or (weights, bias) tuples.
    eval_loader : DataLoader
        Test DataLoader over pre-extracted features.
    model : torch.nn.Module
        The model to overwrite classifier parameters for inference.
    device : str
        Evaluation device."""

    dist_scales = [_[1].detach() for _ in prototype_ensemble]
    clf = torch.concat([_[0] for _ in prototype_ensemble], dim=1).detach().transpose(0, 1)
    preds_list = torch.zeros(clf.shape[0], len(eval_loader.dataset), eval_loader.dataset.num_labels)

    position = 0

    with torch.no_grad():
        for *_, feats in eval_loader:
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

    return preds_list


def _evaluate_ensemble(prototype_ensemble, loader, full_model, device='cuda'):
    """
    Evaluate the prototype ensemble on a dataset.
    Parameters
    ----------
    prototype_ensemble: list
        List of (prototypes, distance_scale) or (weights, bias) tuples.
    loader DataLoader
        DataLoader for the dataset to evaluate on.
    full_model: torch.nn.Module
        The model to overwrite classifier parameters for inference.
    device: str
        Device to run inference on (e.g., 'cuda' or 'cpu').
    Returns
    -------
    dict
        A dictionary of performance metrics including accuracy per group, class, and attribute,
        as computed by `eval_metrics()`.
    """

    ds = loader.dataset
    classes, attributes, groups = np.array(ds.y), np.array(ds._a), np.array(ds.g)

    pred_list = _predict(
        prototype_ensemble,
        loader,
        full_model,
        device=device,
    )
    pred_list = pred_list.softmax(dim=2).mean(0).detach().cpu().numpy()
    res = eval_metrics(pred_list, classes, attributes, groups)

    return res


@dataclass(frozen=True)
class Args:
    data_dir: str = field(default="",
                          metadata={"help": "Path to the directory containing features (e.g., .npy files)."})
    metadata_path: str = field(default="", metadata={"help": "Path to the metadata CSV or JSON file."})
    num_classes: int = field(default=None, metadata={"help": "Number of classes in the dataset."})
    norm_emb: bool = field(default=True, metadata={"help": "Whether to normalize pre-extracted features (per sample)."})

    dataset_name: str = field(default='Features',
                              metadata={"help": "Name of the dataset class to use (e.g., 'Waterbirds')."})
    device: str = field(default='cuda',
                        metadata={"help": "Device to use for training and inference (e.g., 'cuda' or 'cpu')."})
    workers: int = field(default=0, metadata={"help": "Number of DataLoader workers."})

    batch_size_train: int = field(default=256, metadata={"help": "Batch size for training."})
    batch_size_eval: int = field(default=256, metadata={"help": "Batch size for evaluation (val/test)."})

    train_attr: bool = field(default=False, metadata={
        "help": "Use attribute labels (if available) to construct balanced training sets."})
    seed: int = field(default=None, metadata={"help": "Random seed for reproducibility."})

    epochs: int = field(default=20, metadata={"help": "Number of epochs to train each prototype classifier."})
    lr: float = field(default=1e-3, metadata={"help": "Learning rate for optimizer."})

    multi_class: bool = field(default=False, metadata={
        "help": "Enable multi-class (instead of binary) classification using `yy` in metadata."})
    num_stages: int = field(default=15, metadata={"help": "Number of ensemble members (stages) to train."})
    emb_dim: int = field(default=2048, metadata={"help": "Embedding dimension of the backbone features."})

    split_map: dict = field(default=None,
                            metadata={"help": "Mapping from split names (e.g., {'val': 'va', 'test': 'te'})."})
    scheduler: str = field(default='none',
                           metadata={"help": "Learning rate scheduler to use (e.g., 'none', 'onecycle')."})

    cov_reg: float = field(default=5e4, metadata={
        "help": "Initial coefficient for covariance regularization (prototype diversity)."})
    entropic: int = field(default=30, metadata={"help": "Entropic scale for IsoMax loss."})
    show_freq: int = field(default=10, metadata={"help": "Logging frequency (in epochs)."})

    optim: str = field(default='sgd', metadata={"help": "Optimizer to use (either 'sgd' or 'adam')."})
    trn_split: str = field(default='va',
                           metadata={"help": "Data split to use as training set for prototype learning (e.g., 'va')."})
    loss_name: str = field(default='isomax', metadata={"help": "Loss function: 'isomax' or 'ce' (cross-entropy)."})

    subsample_type: str = field(default='group',
                                metadata={"help": "Subsampling strategy: e.g., 'group', 'attribute', or 'none'."})
    alpha: float = field(default=0.05, metadata={"help": "Decay factor for covariance regularization scheduler."})
    eval_freq: int = field(default=-1, metadata={
        "help": "How often to evaluate ensemble online. -1 disables intermediate evaluation."})

    @classmethod
    def help(cls):
        print("Available configuration parameters:\n")
        for f in fields(cls):
            name = f.name
            default = f.default
            help_text = f.metadata.get("help", "")
            print(f"  {name} (default: {default})\n    → {help_text}\n")
