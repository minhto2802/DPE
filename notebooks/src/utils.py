import torch
import torchvision
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from functools import partial

from utils import datasets as dsets
from utils.misc import fix_random_seed
from utils.eval_helpers import eval_metrics, get_acc
from utils.isomaxplus import IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart


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
            outputs = model[-1](feats)  # Assume model is [backbone, ..., head]; use head only
            all_preds.append(outputs.detach().softmax(1).cpu())

        all_preds = torch.concat(all_preds, dim=0).numpy()

        # Compute per-group/class metrics
        res = eval_metrics(all_preds, np.array(classes), np.array(attributes), np.array(groups))

    return res


def get_subsampled_train_set(
        datasets=None,
        data_dir='/scratch/ssd004/scratch/minht/datasets/',
        attr_availability='yes',
        subsample_type='group',
        dataset_name='Waterbirds',
        trn_split='va',
):
    """
    Initialize a subgroup-balanced or attribute-balanced training dataset.

    Parameters
    ----------
    datasets : dict or None
        A dictionary of pre-loaded datasets. If None, a new dict is created.
    data_dir : str
        Root directory of all datasets.
    attr_availability : str
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

    datasets['train'] = vars(dsets)[dataset_name](
        data_dir, trn_split, None,
        train_attr=attr_availability,
        subsample_type=subsample_type,
        pre_extracted_feats=pre_extracted_feats,
    )

    return datasets



def get_train_loader(datasets, attr_availability='yes', batch_size=256, workers=8, dataset_name='Waterbirds'):
    """
    Construct a PyTorch DataLoader for training with subsampled training data.

    Parameters
    ----------
    datasets : dict
        Dictionary of dataset splits (expects at least 'val', and will create 'train').
    attr_availability : str
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
        attr_availability=attr_availability,
        dataset_name=dataset_name
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


def init_model(ckpt_path, num_classes=2, model=None, device='cuda'):
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
                     entropic=30, lr=1e-3, stage=1, verbose=True, loss_name='isomax'):
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
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if len(prototype_ensemble) > 0:
        prototype_ensemble = torch.concat([_[0] for _ in prototype_ensemble], dim=1).detach()

    for epoch in range(epochs):
        model.train()
        running_loss, running_clf, running_cov, running_correct, total = 0.0, 0.0, 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"[Stage {stage}] Epoch {epoch + 1}",
                    leave=(epoch == epochs - 1) and verbose)

        for _, _, labels, _, _, feats in pbar:
            feats = feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model[-1](feats)
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

            preds = outputs.argmax(dim=1)
            correct = (preds == labels).sum().item()
            running_loss += loss.item()
            running_clf += clf_loss.item()
            running_cov += cov_loss.item()
            running_correct += correct
            total += labels.size(0)

            pbar.set_postfix({
                'loss': running_loss / (total // labels.size(0)),
                'clf': running_clf / (total // labels.size(0)),
                'cov': running_cov / (total // labels.size(0)),
                'acc': f"{running_correct / total:.2%}",
                'val wga': f"{val_wga:.2%}"
            })

        val_wga = evaluate(model, val_loader)['min_group']['accuracy']
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


def train_ensemble(init_model_func, datasets, dataloaders, init_train_loader,
                   cov_reg=5e5, random_subset=True,
                   num_stages=15, show_freq=15, seed=0, epochs=20,
                   wd_weight=10, entropic=30, lr=1e-3, loss_name='isomax'):
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
    seed : int
        Random seed.
    epochs : int
        Epochs per stage.
    wd_weight : float
        Prototype norm regularization weight.
    entropic : float
        IsoMax entropic scale.
    lr : float
        Learning rate.
    loss_name : str
        Either 'isomax' or 'ce'.

    Returns
    -------
    tuple
        Final worst-group, balanced, and detailed evaluation results.
    """
    fix_random_seed(seed)
    full_model = None
    prototype_ensemble = []
    ensemble_wga, ensemble_acc, ensemble_balanced_acc = [], [], []
    _train_prototypes = partial(train_prototypes, val_loader=dataloaders['val'],
                                epochs=epochs, wd_weight=wd_weight, device='cuda', entropic=entropic, lr=lr,
                                loss_name=loss_name)
    cv_scheduler = cov_reg_scheduler_inv(cov_reg, num_stages)
    train_loader = init_train_loader(datasets)

    for stage in range(1, num_stages + 1):
        verbose = stage % show_freq == 0 or stage == 1 or stage == num_stages + 1
        full_model = init_model_func(model=full_model)
        prototype_ensemble.append(_train_prototypes(
            train_loader, model=full_model,
            prototype_ensemble=prototype_ensemble, stage=stage, verbose=verbose,
            cov_reg=cv_scheduler(stage),
        ))
        res = evaluate_ensemble(prototype_ensemble, dataloaders['test'], full_model, device='cuda', verbose=verbose)
        ensemble_wga.append(res['min_group']['accuracy'])
        ensemble_acc.append(res['overall']['accuracy'])
        ensemble_balanced_acc.append(res['overall']['balanced_acc'])

        if stage <= num_stages + 1 and random_subset:
            train_loader = init_train_loader(datasets)
    return ensemble_wga, ensemble_balanced_acc, res


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
                preds_list[i][position:position + feats.shape[0]] = model[-1](feats.squeeze())
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
