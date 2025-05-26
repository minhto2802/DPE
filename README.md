# <u>D</u>iverse <u>P</u>rototypical <u>E</u>nsembles Improve Robustness to Subpopulation Shift

![Diverse Prototypical Ensemble Training Pipeline](docs/figures/embeddings_figure.png)

---

Overview
========

This repository contains the official implementation and experiments for our ICML 2025 paper:  
**Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift**  
Project summary site: https://minhto2802.github.io/diversified_prototypical_ensemble

We propose a simple and scalable approach for improving model robustness under subpopulation shift, without relying on explicit group annotations. The method builds on the intuition that diverse classifiers trained on different parts of the data distribution can complement one another, especially when subgroups are not well represented in training.

Our approach combines a pretrained backbone with a *diversified ensemble of prototype-based classifiers*, each trained on a different balanced subset of data. Diversity among ensemble members is further encouraged via an inter-prototype similarity loss, resulting in broader coverage and better generalization to underrepresented subgroups.

The training pipeline includes:

- **Stage-0**: Supervised backbone pretraining using ERM loss.
- **Stage-1+**: Training multiple prototype classifiers on resampled subsets to form an ensemble.

> This framework is designed to be flexible and applicable in both with and without subgroup annotation scenarios.


---

Notebooks
=============================

We provide a collection of Jupyter notebooks under the [`notebooks/`](notebooks/) directory to illustrate key components of Diverse Prototypical Ensembles (DPE) through visualization, controlled experiments, and ablation studies. These notebooks provide a walkthrough of the motivation and implementation of our method as described in the paper, demonstrated on two standard benchmark datasets.

- **[`00_synthetic.ipynb`](notebooks/00_synthetic.ipynb)**  
  A 2D synthetic experiment that simulates subpopulation shift under controlled conditions.  
  This notebook visualizes the limitations of standard classifiers trained on imbalanced subgroups and demonstrates how DPE achieves better coverage and robustness through diversified prototype ensembles.

- **[`01_waterbirds_with_attribute_annotation.ipynb`](notebooks/01_waterbirds_with_attribute_annotation.ipynb)**  
  Full pipeline demonstration of DPE on the Waterbirds dataset, using group-annotated validation data.  
  This notebook highlights the effectiveness of training diverse classifiers on balanced group subsets, and evaluates per-group accuracy improvements over the ERM baseline.

- **[`02_celeba_without_attribute_annotation.ipynb`](notebooks/02_celeba_without_attribute_annotation.ipynb)**  
  Application of DPE to the CelebA dataset in a more realistic setting where subgroup labels are not available.  
  It shows that even without group supervision, DPE outperforms strong baselines such as Deep Feature Reweighting (DFR) in worst-group accuracy. The notebook also illustrates that increasing the number of DFR heads does not further improve fairness, while DPE consistently improves both robustness and subgroup equity.

> Each notebook is self-contained and can be executed independently. These examples serve as a foundation for adapting DPE to other datasets and deployment scenarios.


Reproducing the Paper Results
=============================

This section provides all the steps and configuration options needed to reproduce the experiments from our ICML 2025 paper.

---

## 1. Quickstart

### Stage-0 training (ERM)

Run the following to train a supervised backbone from scratch:

```
sbatch scripts/train.sh \
    --dataset_name Waterbirds \
    --model_name resnet152 \
    --epochs 400
```

### Stage-1+ training (Diversified Prototypes)

Once Stage-0 is complete, launch the prototype ensemble training using the pretrained backbone:

```
sbatch scripts/train_pe.sh \
    --dataset_name Waterbirds \
    --pretrained_path /scratch/.../Waterbirds/*/ckpt_last.pt \
    --epochs 20 \
    --lr 1e-3 \
    --cov_reg 1e5
```

### Launch All Jobs (Predefined Datasets)

To run all supported configurations:

```
sbatch scripts/train_all.sh
sbatch scripts/train_all_pe.sh
```

---

## 2. Key Arguments

### General

- `--dataset_name`: Waterbirds, CelebA, MultiNLI, etc.
- `--model_name`: resnet50, bert-base-uncased
- `--epochs`, `--lr`: training schedule and learning rate
- `--seed`: for reproducibility

### Stage-0

- `--loss_name`: `ce` (default) or `isomax`
- `--train_mode`: `full` (default) or `freeze`

### Stage-1+

- `--stage 1`
- `--pretrained_path`: path to checkpoint from Stage-0
- `--num_stage`: number of ensemble members (default: 16)
- `--cov_reg`: strength of prototype decorrelation regularization
- `--subsample_type`: `group` or `class` (balanced subset type)
- `--entropic_scale`: IsoMax hyperparameter
- `--train_mode freeze`: linear probing (recommended for Stage-1+)
- `-ncbt`: disable class-balanced training
- `-sit`: shuffle training samples at each epoch

---

## 3. Training Tips

- **Monitoring**: In W&B, relevant metrics are logged under keys prefixed with `ensemble_` (e.g., `ensemble_worst_group_acc`).
- **Prototype Regularization**: Tune `--cov_reg` in the range [1e4, 1e6] to control prototype diversity.
- **IsoMax Scaling**: Typical `--entropic_scale` values are between 10 and 40 depending on the dataset.
- **Sampling Behavior**:
  - `--subsample_type group` (with `--train_attribute yes`) performs subgroup-balanced sampling.
  - `--subsample_type class` (or `--train_attribute no`) performs class-balanced sampling.
- **Epochs**: Stage-1+ typically benefits from 15–30 epochs of training.
- **Artifacts**:
  - Checkpoints: `/checkpoint/$USER/$SLURM_JOB_ID/ckpt_*.pt`
  - Logs: `logs/<jobname>.<id>.log`
  - W&B grouping: controlled via `--wdb_group`
- **Debug Mode**: Use `--no_wandb` to disable Weights & Biases logging.

---

## 4. Expected Outputs

### Stage-0

- `ckpt_best_acc.pt`, `ckpt_best_bal_acc.pt`, `ckpt_last.pt` (used in ensemble training)
- `feats_val.npy`, `feats_test.npy` (optional feature dumps)

### Stage-1+

- `prototype_ensemble_<criterion>.pt` (criterion can be `wga_val`, `last`, etc.)
- `dist_scales_<criterion>.pt` (corresponding distance scale parameters)
- Precomputed embeddings (automatically saved to `--ckpt_dir`)
- W&B logs (if enabled)

---

These steps and configurations are consistent with the results reported in our ICML 2025 submission. For further illustration and qualitative experiments, see the [Notebooks](#notebooks) section.


Results
-------------------------

Worst-group accuracy on datasets **without subgroup annotations**:

| Algorithm         | Waterbirds | CelebA   | CivilComments | MultiNLI | MetaShift | CheXpert | ImageNetBG | NICO++   | Living17 |
|-------------------|------------|----------|---------------|----------|-----------|----------|------------|----------|----------|
| ERM*              | 77.9±3.0   | 66.5±2.6 | 69.4±1.2      | 66.5±0.7 | 80.0±0.0  | 75.6±0.4 | 86.4±0.8   | 33.3±0.0 | 53.3±0.9 |
| ERM* + DPE (Ours) | 94.1±0.2   | 84.6±0.8 | 68.9±0.6      | 70.9±0.8 | 83.6±0.9  | 76.8±0.1 | 88.1±0.7   | 50.0±0.0 | 63.0±1.7 |

Worst-group accuracy on datasets **with subgroup annotation**:

| Algorithm         | Group Info<br>(Train / Val) | WATERBIRDS | CELEBA   | CIVILCOMMENTS | MULTINLI | METASHIFT | CHEXPERT |
|-------------------|-----------------------------|------------|----------|---------------|----------|-----------|----------|
| ERM*              | X / X                       | 77.9±3.0   | 66.5±2.6 | 69.4±1.2      | 66.5±0.7 | 80.0±0.0  | 75.6±0.4 |
| ERM* + DPE (ours) | X / ✓✓                      | 94.1±0.4   | 90.3±0.7 | 70.8±0.8      | 75.3±0.5 | 91.7±1.3  | 76.0±0.3 |

✗: no group info is required  
✓: group info is required for hyperparameter tuning  
✓✓: validation data is required for training and hyperparameter tuning

More tables and detailed experimental breakdowns are available at:  
https://github.com/anonymous102030411/anon

---

Citation
--------

```
@inproceedings{to2025dpe,
  title     = {Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift},
  author    = {Nguyen Nhat Minh To and Paul F. R. Wilson and Viet Nguyen and Mohamed Harmanani and Michael Cooper and Fahimeh Fooladgar and Purang Abolmaesumi and Parvin Mousavi and Rahul Krishnan},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```

Acknowledgements
----------------

Much of the training and evaluation infrastructure in this repository was adapted from: https://github.com/YyzHarry/SubpopBench  
IsoMax loss function implementation was provided by https://github.com/dlmacedo/entropic-out-of-distribution-detection

We thank the authors for releasing their well-organized benchmark and codebase.
