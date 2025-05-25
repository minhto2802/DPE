import os
import pprint
from glob import glob
from typing import Optional, Dict, Any

import wandb
import pylab as plt
import seaborn as sns
from tqdm import tqdm
from jsonargparse import ArgumentParser

import torch.nn.functional as F

from utils.misc import *
from utils import datasets as dsets, DATASETS
from utils import IsoMaxPlusLossFirstPart, IsoMaxPlusLossSecondPart
from utils import get_dataloaders, get_model, eval_metrics, get_acc, log_wandb, bert_adamw_optimizer, timer

torch.set_warn_always(False)

INF = 99999


def get_args():
    parser = ArgumentParser(env_prefix='', default_env=True, logger=True, print_config='--print_config')

    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training.')
    parser.add_argument('--batch_size_eval', default=512, type=int, help='Batch size for evaluation.')

    parser.add_argument('--workers', default=12, type=int, help='Number of data loader workers.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--device', default='cuda', type=str, help='Device to train on (e.g., "cuda", "cpu").')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs.')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for SGD.')
    parser.add_argument('--wd_weight', default=10, type=float,
                        help='Weight decay coefficient for prototype regularization.')
    parser.add_argument('--optim', default='sgd', type=str, choices=['sgd', 'adam', 'bert_adam'],
                        help='Optimizer choice.')
    parser.add_argument('--optim_weight_decay', default=0, type=float, help='Weight decay for optimizer.')
    parser.add_argument('--scheduler', choices=['none', 'onecycle', 'bert'], default='none',
                        help='Learning rate scheduler.')

    parser.add_argument('--cov_reg', default=1e5, type=float,
                        help='Regularization strength for covariance-based diversity.')
    parser.add_argument('--dfr_reg', default=0.1, type=float, help='L1 regularization strength on classifier weight.')

    parser.add_argument('--dataset_name', default='Waterbirds', type=str, choices=DATASETS, help='Dataset to use.')
    parser.add_argument('--stage', default=0, type=int, help='Starting training stage (for ensembles).')
    parser.add_argument('--num_stages', default=1, type=int, help='Number of training stages (ensemble members).')
    parser.add_argument('--subsample_target', default=None, type=none_or_str, help='Subsample target group.')
    parser.add_argument('--filter_perc', type=int, default=0, help='Percentage of training samples to filter.')
    parser.add_argument('--num_samples', type=int, default=None, help='Total number of training samples to use.')

    parser.add_argument('-ncbt', '--no_class_balanced_training', action='store_true',
                        help='Disable class-balanced training.')
    parser.add_argument('--no_augmentation', action='store_true', help='Disable data augmentation.')
    parser.add_argument('--dynamic_num_samples', action='store_true', help='Enable dynamic sampling per epoch.')
    parser.add_argument('-sit', '--shuffle_in_training', action='store_true', help='Shuffle data in training set.')
    parser.add_argument('--train_attr', type=str, default='no', choices=['yes', 'no'],
                        help='Use attribute annotations during training.')
    parser.add_argument('--norm_emb', type=str, default='yes', choices=['yes', 'no'],
                        help='Normalize feature embeddings.')
    parser.add_argument('--data_dir', type=str, default='/scratch/ssd004/scratch/minht/datasets',
                        help='Root directory for dataset.')
    parser.add_argument('--text_arch', type=str, default='bert-base-uncased', help='Text architecture for NLP tasks.')
    parser.add_argument('--subsample_type', type=str, default=None, help='Subsampling strategy to use.')
    parser.add_argument('--eval_freq', type=int, default=1, help='Evaluation frequency (in epochs).')
    parser.add_argument('--fix_training_set', action='store_true', help='Fix training set across ensemble stages.')

    parser.add_argument('--model_name', default='resnet50', type=str, help='Backbone model name.')
    parser.add_argument('--pretrained_path', default=None, type=str, help='Path to pretrained model checkpoint.')
    parser.add_argument('--pretrained_imgnet', action='store_true', help='Use pretrained weights inside training.')
    parser.add_argument('--force_saving_feats', action='store_true', help='Force re-saving extracted features.')

    parser.add_argument('--train_mode', type=str, default='full', choices=['full', 'freeze'],
                        help='Training mode: full fine-tuning or freeze backbone.')
    parser.add_argument('--loss_name', type=str, default='ce', choices=['ce', 'isomax'], help='Loss function to use.')
    parser.add_argument('-ec', '--ensemble_criterion', type=str, choices=['wga_val', 'wga_test', 'last'],
                        default='wga_val', help='Criterion for ensemble selection.')
    parser.add_argument('-es', '--entropic_scale', default=10, type=float, help='Entropic scaling for IsoMax loss.')

    parser.add_argument('--verbose', action='store_true', help='Print verbose logs.')
    parser.add_argument('--no_wandb', action='store_true', help='Disable Weights & Biases logging.')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode.')
    parser.add_argument('--suffix', type=str, default='', help='Suffix to add to run name.')
    parser.add_argument('--wdb_group', type=str, default=None, help='W&B group identifier for organizing runs.')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Directory to save checkpoints.')

    args = parser.parse_args()
    return args


def eval_model(
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: str = 'cuda',
        return_feats: bool = False,
) -> tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
    """
    Evaluates a model on a given dataloader and computes predictions, losses,
    and optionally feature embeddings.

    Args:
        model (torch.nn.Module): The model to be evaluated. Assumed to have either `.fc` or `[-1]` as classifier head.
        dataloader (torch.utils.data.DataLoader): DataLoader providing input batches for evaluation.
        criterion (torch.nn.Module): Loss function used to compute evaluation loss.
        device (str): Device on which to perform computation (default: 'cuda').
        return_feats (bool): Whether to return intermediate feature representations from the backbone.

    Returns:
        tuple:
            - feats (Optional[np.ndarray]): Extracted feature embeddings if `return_feats=True`, otherwise None.
            - preds (np.ndarray): Softmax probabilities of model predictions.
            - losses (np.ndarray): Per-batch loss values computed using `criterion`.
    """

    model.eval()
    backbone = torch.nn.Sequential(*list(model.children())[:-1])
    feats, preds, losses = [], [], []

    with torch.no_grad():
        for _, inputs, labels, *_ in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            _feats = backbone(inputs).squeeze()
            outputs = model.fc(_feats) if hasattr(model, 'fc') else model[-1](_feats)

            loss = criterion(outputs, labels)
            _preds = F.softmax(outputs, -1)

            if return_feats:
                feats.append(_feats.detach().cpu())
            preds.append(_preds.detach().cpu())
            losses.append(loss.detach().cpu())

    if len(feats):
        feats = torch.concatenate(feats).numpy()
    preds = torch.concatenate(preds).numpy()
    losses = torch.concatenate(losses).numpy()
    return feats, preds, losses


@timer
def train_model(
        args: Any,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        dataloaders: Dict[str, torch.utils.data.DataLoader],
        num_epochs: int = 25,
        device: str = 'cuda',
        ckpt_path: Optional[str] = None,
        prototypes_ensemble: Optional[torch.Tensor] = None,
        stage: int = 0,
        train_mode: str = 'full',
        skipped_phase: Optional[list] = None,
        run: Any = DummyRun(),
        wd_weight: float = 10,
        worst_val_metrics: Optional[Dict[str, list]] = None,
        ensemble_bw: Optional[list] = None,
        worst_metric: str = 'recall',
        cov_reg: float = 1e4,
) -> tuple[torch.nn.Module, dict, dict, Optional[list]]:
    """
    Trains a neural network model using staged training with support for prototypical ensembles
    and evaluation under subpopulation shift conditions.

    Args:
        args (Any): Parsed configuration object containing all hyperparameters and flags.
        model (torch.nn.Module): The neural network to be trained.
        criterion (torch.nn.Module): Loss function used for optimization.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        scheduler (Optional[Any]): Learning rate scheduler, applied per epoch.
        dataloaders (Dict[str, DataLoader]): Dataloaders for 'train', 'val', and 'test' splits.
        num_epochs (int): Number of training epochs.
        device (str): Device on which computations will be performed ('cuda' or 'cpu').
        ckpt_path (Optional[str]): Path to save model checkpoints.
        prototypes_ensemble (Optional[torch.Tensor]): Ensemble prototypes for covariance regularization.
        stage (int): Current training stage (used in multi-stage ensemble training).
        train_mode (str): Either 'full' (fine-tuning) or 'freeze' (linear probing).
        skipped_phase (Optional[list]): List of phases to skip ('val', 'test', etc.).
        run (Any): Weights & Biases or compatible logging object.
        wd_weight (float): weight decay for the prototypes.
        worst_val_metrics (Optional[Dict[str, list]]): Metric tracking dictionary for worst-group accuracy.
        ensemble_bw (Optional[list]): Accumulator list for ensemble classifiers selected by val performance.
        worst_metric (str): Metric name used to identify worst-group condition (e.g., 'recall').
        cov_reg (float): Coefficient for covariance-based regularization across ensemble prototypes.

    Returns:
        tuple: (model, results dict, worst_val_metrics, ensemble_bw)
    """

    results = {}
    metric_columns = ['balanced_acc', 'accuracy', 'AUROC']

    best_acc = best_bal_acc = best_worst_metric = 0.0
    step, end_training = 1, False

    for epoch in range(num_epochs):
        metrics = None
        print(f'[Stage {stage}] Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'test']:
            if (phase not in dataloaders.keys()) or (skipped_phase is not None and phase in skipped_phase):
                continue

            if phase == 'train' and train_mode == 'full':  # so that BN stats won't be updated
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            n_instances = running_loss = running_loss_clf = running_loss_cov = running_corrects = 0.0
            all_preds = []

            # Iterate over data.
            for _, inputs, labels, _, _, feats in dataloaders[phase]:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                cov_loss = torch.tensor(0.0)

                with torch.set_grad_enabled(phase == 'train'):
                    if stage > 0:
                        feats = feats.to(device)
                        outputs = model[-1](feats)
                        clf_loss = criterion(outputs, labels)
                        if phase == 'train':
                            if isinstance(criterion, IsoMaxPlusLossSecondPart):
                                head = model[-1]
                                n_classes = head.prototypes.shape[0]
                                wd = torch.einsum('ijk,ilk->ijl',
                                                  [head.prototypes[:, None],
                                                   head.prototypes[:, None]]) * wd_weight
                                wd = wd.squeeze().mean()
                                loss = clf_loss + wd
                                if (prototypes_ensemble is not None) and (cov_reg > 0):
                                    _prototypes = torch.concat([head.prototypes[:, None], prototypes_ensemble], dim=1)
                                    with torch.set_grad_enabled(cov_reg > 0):
                                        n_pro, n_dim = _prototypes.shape[1:]
                                        cov = torch.einsum('ijk,ilk->ijl', [_prototypes, _prototypes]) / (n_dim - 1)
                                        cov_loss = torch.abs(cov[:, 0, 1:].sum(1).div(n_pro).mean())
                                        if cov_reg:
                                            loss = loss + cov_loss * cov_reg
                                else:
                                    weight = model[-1].weight  # if args.loss_name == 'ce' else model[-1].prototypes
                                    loss = clf_loss + args.dfr_reg * torch.norm(weight, 1)
                        else:
                            loss = clf_loss

                    else:
                        outputs = model(inputs)
                        loss = clf_loss = criterion(outputs, labels)

                    if phase != 'train':
                        all_preds.append(outputs.detach().softmax(1).cpu())

                    _, preds = torch.max(outputs, 1)
                    n_instances += len(inputs)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_loss_clf += clf_loss.item() * inputs.size(0)
                running_loss_cov += cov_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                step += 1
                if num_epochs == INF and (step == dataloaders['train'].dataset.N_STEPS):
                    end_training = True
                    break

            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()

            epoch_loss = running_loss / n_instances
            epoch_acc = running_corrects.double() / n_instances
            results[phase] = {'loss': epoch_loss, 'acc': epoch_acc}
            if stage == 0:
                run.log({f'loss/{phase}': epoch_loss}, commit=False)
                run.log({f'ovr_acc/{phase}': epoch_acc}, commit=False)

            if phase == 'train':
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                print(f'train CLF Loss: {running_loss_clf / n_instances:.6f}')
                print(f'train Cov Loss: {running_loss_cov / n_instances:.6f}')
                if stage == 0:
                    run.log({'running_loss_clf': running_loss_clf / n_instances}, commit=False)
                    run.log({'running_loss_cov': running_loss_cov}, commit=False)

            # Evaluation on validation set
            else:
                ds = dataloaders[phase].dataset
                all_preds = torch.concat(all_preds, dim=0).numpy()
                res = eval_metrics(all_preds, np.array(ds.y), np.array(ds._a), np.array(ds.g))

                if max(ds.y) < 10:
                    print('Per group ACC: ',
                          list(np.around([res['per_group'][i]['accuracy'] * 100 for i in res['per_group'].keys()], 2)))
                    print('Per class F1-Score: ',
                          list(np.around([res['per_class'][i]['f1-score'] * 100 for i in res['per_class'].keys()], 2)))
                if metrics is None:
                    metrics = pd.DataFrame({k: 0. for k in metric_columns}, index=[phase])
                metrics.loc[phase] = [np.round(res['overall'][k] * 100, 2) for k in metric_columns]
                log_wandb(run, phase, res)

                if worst_val_metrics is None:
                    worst_val_metrics = {}
                k = f'wga_{phase}'
                if k not in worst_val_metrics.keys():
                    worst_val_metrics[k] = [res['min_group']['accuracy']]
                else:
                    worst_val_metrics[k].append(res['min_group']['accuracy'])

                # Validation phase metrics
                if epoch_acc > best_acc and ckpt_path:
                    best_acc = float(epoch_acc)
                    if stage == 0:
                        torch.save(model.state_dict(), f"{ckpt_path}/ckpt_best_acc.pt")
                epoch_bal_acc = res['overall']['balanced_acc']
                if epoch_bal_acc > best_bal_acc and ckpt_path:
                    best_bal_acc = float(epoch_bal_acc)
                    if stage == 0:
                        torch.save(model.state_dict(), f"{ckpt_path}/ckpt_best_bal_acc.pt")

                if worst_val_metrics is None:
                    worst_val_metrics = {}
                for k in res['per_class'][0].keys():
                    tmp = []
                    if k in ['support']:
                        continue
                    for c in res['per_class'].keys():
                        tmp.append(res['per_class'][c][k])
                    if k not in worst_val_metrics.keys():
                        worst_val_metrics[k] = [np.min(tmp)]
                    else:
                        worst_val_metrics[k].append(np.min(tmp))
                    if stage == 0:
                        run.log({f'worst_val_metrics/{k}': worst_val_metrics[k][-1]}, commit=False)

                if worst_val_metrics[worst_metric][-1] >= best_worst_metric:
                    best_worst_metric = worst_val_metrics[worst_metric][-1]
                    if stage == 0:
                        torch.save(model.state_dict(), f"{ckpt_path}/ckpt_best_{worst_metric}.pt")
                    if ensemble_bw is not None:
                        assert isinstance(ensemble_bw, list)
                        classifier = _extract_classifier(args.loss_name, model)
                        if len(ensemble_bw) < stage:
                            ensemble_bw.append(classifier)
                        else:
                            ensemble_bw[-1] = classifier

        if stage == 0:
            run.log({'stage': stage, 'epoch': epoch})

        print(f'Best val Acc: {best_acc:4f}')

        if end_training:
            break

    if ckpt_path is not None and (stage == 0):
        torch.save(model.state_dict(), f"{ckpt_path}/ckpt_last.pt")

    return model, results, worst_val_metrics, ensemble_bw


def _extract_classifier(loss_name, model):
    if loss_name == 'ce':
        classifier = [model[-1].weight[:, None].detach().clone()]
    else:
        classifier = [
            model[-1].prototypes[:, None].detach().clone(),
            model[-1].distance_scale.detach().clone()
        ]
    return classifier


@timer
def evaluate_ensemble_fixed_backbone(
        ensemble: torch.Tensor,
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        distance_scales: Optional[list[torch.Tensor]] = None,
        run: Any = DummyRun(),
        phase: str = 'test',
        return_preds: bool = False,
        norm_emb: str = 'yes'
) -> Optional[torch.Tensor]:
    """
    Evaluates an ensemble of classifiers with a fixed backbone using pre-extracted or computed features.

    Args:
        ensemble (torch.Tensor): A tensor of shape (num_classes, num_ensemble_members * feature_dim), transposed inside the function.
        dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation set, expected to yield tuples including features.
        model (torch.nn.Module): A backbone + classifier model where the classifier head can be replaced dynamically.
        distance_scales (Optional[List[torch.Tensor]]): Optional per-member distance scaling factors (used for IsoMax-based models).
        run (Any): W&B or dummy logger object with `.log()` API for tracking evaluation metrics.
        phase (str): Phase name (e.g., 'test', 'val') used for logging and print context.
        return_preds (bool): If True, returns the full tensor of predictions from all ensemble members.
        norm_emb (str): Whether to normalize feature embeddings ('yes' or 'no').

    Returns:
        Optional[torch.Tensor]: Returns the ensemble predictions tensor of shape (ensemble_size, num_samples, num_classes) if `return_preds=True`; otherwise returns None.
    """
    print(f'[{phase}]')
    ensemble = ensemble.transpose(0, 1).to('cuda')
    preds_list = torch.zeros(ensemble.shape[0], len(dataloader.dataset), dataloader.dataset.num_labels)
    model.eval()
    position = 0

    ds = dataloader.dataset
    classes, attributes, groups = np.array(ds.y), np.array(ds._a), np.array(ds.g)

    with torch.no_grad():
        for _, x, y, _, g, feats in tqdm(dataloader):
            if feats.ndim == 1:
                feats = model[:-1](x.to('cuda'))
                if norm_emb == 'yes':
                    feats = (feats - feats.mean(dim=1, keepdims=True)) / feats.std(dim=1, keepdims=True)
            else:
                feats = feats.to('cuda')

            for i, weight in enumerate(ensemble):
                if distance_scales is not None:
                    model[-1].prototypes = torch.nn.Parameter(weight, requires_grad=False)
                    model[-1].distance_scale = nn.Parameter(distance_scales[i], requires_grad=False)
                else:
                    model[-1].weight = torch.nn.Parameter(weight, requires_grad=False)
                model.eval()
                preds_list[i][position:position + feats.shape[0]] = model[-1](feats.squeeze())
            position += feats.shape[0]

    for i in range(preds_list.shape[0] - 1, -1, -1):
        preds = preds_list[i].softmax(1).argmax(1).numpy()
        get_acc(preds, classes, groups)

    if preds_list[-1].ndim == 2:
        preds = preds_list.softmax(2).mean(0).detach().cpu().numpy()
        res = eval_metrics(preds, classes, attributes, groups)
        log_wandb(run, f'ensemble_{phase}_avg', res, prefix='ensemble_')

    print('Averaging Ensemble')
    get_acc(preds_list.softmax(2).mean(0).argmax(1).numpy(), classes, groups)

    if return_preds:
        return preds_list.detach().cpu()

    return None


def extract_features(
        args: Any,
        dataset: torch.utils.data.Dataset
) -> np.ndarray:
    """
    Extracts features from a dataset using a pretrained model.

    Args:
        args (Any): Configuration object with fields like `dataset_name`, `train_mode`, `device`, etc.
        dataset (torch.utils.data.Dataset): Dataset object from which to extract features.

    Returns:
        np.ndarray: Extracted feature embeddings for the dataset.
    """

    from torch.utils.data import DataLoader

    model = get_model(args.dataset_name, dataset.num_labels, args.train_mode, args.pretrained_path,
                      loss_name='ce', model=None, pretrained_imgnet=False, model_name=args.model_name)
    model.to(args.device)

    dataloader = DataLoader(dataset=dataset, num_workers=args.workers, batch_size=args.batch_size_eval,
                            shuffle=False, pin_memory=False)

    criterion = get_criterion('ce', reduction='none')
    feats, preds, losses = eval_model(model, dataloader, criterion, return_feats=True)

    return feats


def get_pre_extracted_features(
        args: Any,
        dataset: torch.utils.data.Dataset,
        set_name: str,
        pre_extracted_feats: Optional[np.ndarray] = None,
        force_saving: bool = False
) -> np.ndarray:
    """
    Loads or extracts and saves feature embeddings for a dataset split.

    Args:
        args (Any): Configuration object including `ckpt_dir`, `pretrained_path`, `norm_emb`, etc.
        dataset (torch.utils.data.Dataset): Dataset from which to extract or load features.
        set_name (str): Identifier for the dataset split (e.g., 'train', 'val', 'test').
        pre_extracted_feats (Optional[np.ndarray]): Optionally preloaded features. If None, features will be loaded or computed.
        force_saving (bool): If True, recomputes and overwrites saved features even if cached file exists.

    Returns:
        np.ndarray: Normalized feature embeddings for the specified dataset split.
    """
    if pre_extracted_feats is None:
        if args.pretrained_path is not None:
            pretrained_path = glob(args.pretrained_path)
            assert len(pretrained_path) == 1
            pre_extracted_path = f'{os.path.dirname(pretrained_path[0])}/feats_{set_name}.npy'
        else:
            pre_extracted_path = f'{args.ckpt_dir}/feats_{set_name}.npy'

        if (os.path.exists(pre_extracted_path) == 1) and (not force_saving):
            pre_extracted_feats = np.load(pre_extracted_path, mmap_mode='r')
        else:
            pre_extracted_feats = extract_features(args, dataset=dataset)
            np.save(pre_extracted_path, pre_extracted_feats)
        if args.norm_emb:
            pre_extracted_feats = ((pre_extracted_feats - pre_extracted_feats.mean(axis=1, keepdims=True)) /
                                   pre_extracted_feats.std(axis=1, keepdims=True))
    return pre_extracted_feats


def main(args):
    if not args.no_wandb:
        # Log in to your W&B account
        wandb.login()
        # init wandb using config and experiment name
        name = args.dataset_name + args.suffix
        run = wandb.init(
            config=vars(args),
            project='diversified-prototypical-ensemble',
            group=f'{args.wdb_group}',
            dir=args.ckpt_dir,
            name=name,
            resume='allow',
            mode='disabled' if args.no_wandb else 'online',
        )
        wandb.define_metric('ensemble_stage')
        wandb.define_metric('ensemble_*', step_metric="ensemble_stage")
    else:
        run = DummyRun()

    pprint.PrettyPrinter(indent=4).pprint(args.as_dict())

    fix_random_seed(args.seed, True, True)
    prototype_ensemble = None

    pretrained_path = args.pretrained_path
    model = None
    worst_val_metrics, ensemble_bw = None, []
    prototype_ensemble_last = []

    datasets = dict()
    datasets['val'] = vars(dsets)[args.dataset_name](args.data_dir, 'va', args)
    datasets['test'] = vars(dsets)[args.dataset_name](args.data_dir, 'te', args)
    pprint.PrettyPrinter(indent=4).pprint(datasets)
    worst_metric = 'wga_val'
    train_mode = args.train_mode
    epochs = args.epochs if args.epochs > 0 else INF
    subsample_type = args.subsample_type
    pre_extracted_feats, pre_extracted_feats_test = None, None

    for stage in range(args.stage, args.num_stages):
        if args.stage == 0:
            datasets['train'] = vars(dsets)[args.dataset_name](args.data_dir, 'tr', args, train_attr='no',
                                                               augmentation=not args.no_augmentation)
            print(np.unique(np.array(datasets['train'].g)[datasets['train'].idx], return_counts=True)[1])
        else:
            trn_split = 'va'
            for set_name in ['val', 'test']:
                datasets[set_name].feats = get_pre_extracted_features(
                    args, datasets[set_name], set_name,
                    pre_extracted_feats,
                    force_saving=args.force_saving_feats and (args.stage == stage))
                print(f'{set_name.upper()} features are loaded/extracted.')

            if (not args.fix_training_set) or (stage == args.stage):
                datasets['train'] = vars(dsets)[args.dataset_name](
                    args.data_dir, trn_split, args, train_attr=args.train_attr, subsample_type=subsample_type,  # va
                    augmentation=False, stage=stage, pre_extracted_feats=datasets['val'].feats,
                    dynamic_num_samples=args.dynamic_num_samples)
                print(np.unique(np.array(datasets['train'].g)[datasets['train'].idx], return_counts=True)[1])
                print(args.subsample_type, len(datasets['train']))

        dataloaders = get_dataloaders(
            datasets, args.batch_size, args.batch_size_eval, args.workers,
            args.stage, args.no_class_balanced_training, args.shuffle_in_training)
        for k, v in dataloaders.items():
            print(f'[{k}] n steps: ', len(v))

        model = get_model(
            args.dataset_name, datasets['train'].num_labels, train_mode,
            pretrained_path, loss_name=args.loss_name, model=model,
            pretrained_imgnet=args.pretrained_imgnet if stage == 0 else False)
        model.to(args.device)

        criterion = get_criterion(args.loss_name, entropic_scale=args.entropic_scale)

        match args.optim:
            case 'sgd':
                optimizer_ft = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                               weight_decay=args.optim_weight_decay)
            case 'adam':
                optimizer_ft = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                                 weight_decay=args.optim_weight_decay)
            case 'bert_adam':
                optimizer_ft = bert_adamw_optimizer(model, lr=args.lr,
                                                    momentum=0.9,
                                                    weight_decay=args.optim_weight_decay)

        exp_lr_scheduler = get_scheduler_func(
            args.scheduler, args.lr, args.epochs, len(dataloaders['train']))(optimizer_ft)

        if args.ensemble_criterion == 'wga_val':  # 'wga_val' wga_test:
            ensemble_bw = ensemble_bw if stage > 0 else None
        else:
            ensemble_bw = None

        model, _, worst_val_metrics, ensemble_bw = train_model(
            args, model, criterion, optimizer_ft, exp_lr_scheduler, dataloaders,
            num_epochs=epochs, device=args.device, ckpt_path=args.ckpt_dir,
            prototypes_ensemble=prototype_ensemble, stage=stage, train_mode=args.train_mode,
            run=run, wd_weight=args.wd_weight, worst_val_metrics=worst_val_metrics,
            ensemble_bw=ensemble_bw,
            worst_metric=worst_metric, cov_reg=args.cov_reg)

        if args.train_mode == 'freeze':  # Fix the current backbone
            pretrained_path = None

        if (stage > 0) and (args.ensemble_criterion == 'last'):
            classifier = _extract_classifier(args.loss_name, model)
            prototype_ensemble_last.append(classifier)

        ensemble_dicts = {
            worst_metric: ensemble_bw,
            'last': prototype_ensemble_last,
        }
        for i, (k, ensemble) in enumerate(ensemble_dicts.items()):
            if (ensemble is not None) and (len(ensemble) > 0):
                assert isinstance(ensemble, list)

                dist_scales = [_[1].detach() for _ in ensemble] if args.loss_name == 'isomax' else None
                ens = torch.concat([_[0] for _ in ensemble], dim=1).detach()

                print(f'Evaluating ensemble {k}')
                evaluate_ensemble_fixed_backbone(
                    ens, dataloaders['test'], model, distance_scales=dist_scales, run=run, phase=f'test_{k}',
                    norm_emb=args.norm_emb)
                torch.save(ens, f"{args.ckpt_dir}/prototype_ensemble_{k}.pt")
                if args.loss_name == 'isomax':
                    torch.save(dist_scales, f"{args.ckpt_dir}/dist_scales_{k}.pt")

                if k == args.ensemble_criterion:
                    prototype_ensemble = ens.clone()

        run.log({'ensemble_stage': stage})

    if not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    main(get_args())
