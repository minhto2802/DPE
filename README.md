# <u>D</u>iverse <u>P</u>rototypical <u>E</u>nsembles Improve Robustness to Subpopulation Shift

This repository provides the official implementation and experiments for our ICML 2025 paper:  
*Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift*  
by [Minh To](https://minhto2802.github.io/), [Paul F. R. Wilson](https://github.com/pfrwilson), and co-authors.

<p align="center">
  <img src="https://raw.githubusercontent.com/minhto2802/dpe4subpop/main/docs/figures/logo/ubc_logo_white_bg.png" alt="UBC Logo" height="80"/>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/minhto2802/dpe4subpop/main/docs/figures/logo/vector_logo_white_bg.png" alt="Vector Institute Logo" height="80"/>
</p>


<p align="center">
  <a href="https://arxiv.org/abs/2505.23027">
    <img alt="Paper" src="https://img.shields.io/badge/arXiv-2505.23027-B31B1B.svg?style=for-the-badge&logo=arxiv&logoColor=white">
  </a>
  <a href="https://minhto2802.github.io/dpe4subpop/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-DPE4Subpop-blue?style=for-the-badge">
  </a>
  <a href="https://openreview.net/forum?id=qUTiOeM57J">
    <img alt="OpenReview" src="https://img.shields.io/badge/OpenReview-ICML--2025-green?style=for-the-badge">
  </a>
</p>

---

Overview
========

![Diverse Prototypical Ensemble Training Pipeline](docs/figures/embeddings_figure.png)

Machine learning models often experience significant performance degradation when deployed under distribution shifts. A
particularly important and challenging case is **subpopulation shift**, where the proportions of subgroups vary between
training and deployment. Subpopulation shifts occurred in many forms, including spurious correlations, attribute or
class
imbalance, and previously unseen attribute combinations at test time, can lead to large disparities in model
performance across subgroups.

Existing approaches typically modify empirical risk minimization (ERM) using reweighting or group-aware strategies.
However, these often rely on prior knowledge of subgroup structure or annotated group membership, which may not be
available in practice.

We propose **Diverse Prototypical Ensembles (DPE)**, a simple and scalable framework that improves model robustness to
subpopulation shifts **without requiring group annotations**. DPE replaces the standard linear classification head with
an ensemble of *prototype-based classifiers*, each trained on a different balanced subset of data. Diversity is promoted
through an **inter-prototype similarity loss**, encouraging each classifier to attend to different regions of the
feature space.

---

Quickstart
==========
```python
import sys
sys.path.append('..')

from src.dpe import DPE
from src.dpe.misc import fix_random_seed

def main():
    fix_random_seed(0)

    dpe = DPE(
        data_dir='/path/to/pre_extracted/features',  # Directory containing feats_val.npy and feats_test.npy
        metadata_path='/path/to/metadata/metadata.csv',  # CSV with columns: split, y, g
        verbose=True,
        num_stages=5,       # Number of prototype classifiers to train
        device='cuda',      # Set to 'cpu' or 'cuda' depending on hardware
        emb_dim=2048,       # D: feature embedding dimension (expected feature shape: N x D)
    )
    dpe.fit()

if __name__ == '__main__':
    main()
```

👉 **See the full list of available arguments** in `src/dpe/core.py`, inside the `Args` class. The full demonstration can also be found in [`03_demo.ipynb`](notebooks/03_demo.ipynb)

---

Notebooks
===

We provide a collection of Jupyter notebooks under the [`notebooks/`](notebooks/) directory to illustrate key components
of Diverse Prototypical Ensembles (DPE) through visualization, controlled experiments, and ablation studies. These
notebooks provide a walkthrough of the motivation and implementation of our method as described in the paper,
demonstrated on two standard benchmark datasets.

- **[`00_synthetic.ipynb`](notebooks/00_synthetic.ipynb)**  
  A 2D synthetic experiment that simulates subpopulation shift under controlled conditions.  
  This notebook visualizes the limitations of standard classifiers trained on imbalanced subgroups and demonstrates how
  DPE achieves better coverage and robustness through diversified prototype ensembles.

- **[`01_waterbirds_with_attribute_annotation.ipynb`](notebooks/01_waterbirds_with_attribute_annotation.ipynb)**  
  Full pipeline demonstration of DPE on the Waterbirds dataset, using group-annotated validation data.  
  This notebook highlights the effectiveness of training diverse classifiers on balanced group subsets, and evaluates
  per-group accuracy improvements over the ERM baseline.

- **[`02_celeba_without_attribute_annotation.ipynb`](notebooks/02_celeba_without_attribute_annotation.ipynb)**  
  Application of DPE to the CelebA dataset in a more realistic setting where subgroup labels are not available.  
  It shows that even without group supervision, DPE outperforms strong baselines such as Deep Feature Reweighting (DFR)
  in worst-group accuracy. The notebook also illustrates that increasing the number of DFR heads does not further
  improve fairness, while DPE consistently improves both robustness and subgroup equity.

- **[`03_demo.ipynb`](notebooks/03_demo.ipynb)**
A streamlined demonstration of the DPE training and evaluation workflow using the `dpe` package.  
This notebook serves as a minimal working example to illustrate the integration of DPE into an applied training loop on the Waterbirds dataset:

> Each notebook is self-contained and can be executed independently. These examples serve as a foundation for adapting
> DPE to other datasets and deployment scenarios.

---

Reproducing the Paper Results
=============================

This section provides the steps and configuration details needed to reproduce the experiments from our ICML 2025 paper.

## Data Preparation

We follow the dataset setup instructions from [SubpopBench](https://github.com/YyzHarry/SubpopBench), which provides
scripts and guidelines for preparing all datasets used in our experiments (e.g., Waterbirds, CelebA, MetaShift,
MultiNLI).

To prepare the data:

1. Follow the instructions in the SubpopBench repository to download and preprocess each dataset.
2. Make sure the processed datasets are stored under a common root directory (e.g., `/datasets`).
3. Set `--data_dir` to this root directory when running the training scripts.

## Training Pipeline

- **Stage-0**: Supervised backbone pretraining (ERM or IsoMax).
- **Stage-1+**: Diverse prototype ensemble training on balanced resampled subsets.

> This framework works both with and without access to subgroup annotations.

### Stage-0 Training (ERM)

To fine-tune an ImageNet-pretrained ResNet-50 on the MetaShift dataset (located at `/datasets/metashift`), run:

```
python main.py \
  --epochs 100 \
  --loss_name ce \
  --dataset_name MetaShift \
  --pretrained_imgnet \
  --ckpt_dir /checkpoint/ \
  --data_dir /datasets
```

### Stage-1+ Training (Diversified Prototypes)

Once Stage-0 is complete, initiate prototype ensemble training using the pretrained backbone:

```
python main.py \
  --dataset_name MetaShift \
  --pretrained_path /checkpoint/ckpt_last.pt \
  --ckpt_dir /checkpoint \
  --loss_name isomax \
  --stage 1 \
  --num_stages 16 \
  --epochs 20 \
  --cov_reg 1.e5 \
  --batch-size 64 \
  --optim sgd \
  --lr 1.e-3 \
  --train_attr yes \
  --train_mode freeze \
  --subsample_type group \
  --ensemble_criterion wga_val \
  --entropic_scale 20 \
  -ncbt \
  -sit \
```

### Launch All Predefined Jobs

To run all supported configurations for available datasets:

```
sbatch scripts/train_all.sh
sbatch scripts/train_all_pe.sh
```

## Key Arguments

### General

- `--dataset_name`: e.g., Waterbirds, CelebA, MultiNLI, MetaShift
- `--model_name`: e.g., resnet50, bert-base-uncased
- `--epochs`, `--lr`: controls training length and learning rate
- `--seed`: sets random seed for reproducibility

### Stage-0

- `--loss_name`: `ce` (default)
- `--train_mode`: `full` (default) or `freeze`

### Stage-1+

- `--stage 1`
- `--pretrained_path`: path to Stage-0 model checkpoint
- `--num_stages`: number of ensemble heads (default: 16)
- `--cov_reg`: strength of inter-prototype similarity penalty
- `--subsample_type`: `None` or `group` (group-balanced subsampling if `--train_attr yes` or class-balanced subsampling
  if `--train_attr no`)
- `--entropic_scale`: IsoMax temperature scaling factor
- `--train_mode freeze`: freeze backbone, train only prototypes
- `-ncbt`: disables class-balanced batch construction
- `-sit`: enables data shuffling at each epoch
- `--ensemble_criterion`: ensemble member selection criterion (e.g. `val_wga`: based on the best worst group accuracy on
  the validation set)

## Training Tips

- **Metric Logging**: W&B logs all ensemble-level metrics under the `ensemble_` prefix, such as
  `ensemble_worst_group_acc`.
- **Covariance Regularization**: Tune `--cov_reg` between 1e4 and 1e6 to control prototype diversity.
- **IsoMax Temperature**: Use `--entropic_scale` between 10 and 40 depending on dataset.
- **Balanced Sampling**:
    - `--subsample_type group` ensures subgroup-balanced training when `--train_attr yes`.
    - `--subsample_type class` enables class-balanced sampling when `--train_attr no`.
- **Training Schedule**:
    - Stage-1+ typically converges within 15–30 epochs.
- **Output Directory Layout**:
    - Checkpoints: `/checkpoint/$USER/$SLURM_JOB_ID/ckpt_*.pt`
    - Logs: `logs/<jobname>.<id>.log`
- **Disabling W&B**: Use `--no_wandb` to turn off logging for debugging.

## Expected Outputs

### Stage-0

- Model checkpoints:  
  `ckpt_best_acc.pt`, `ckpt_best_bal_acc.pt`, `ckpt_last.pt`
- Optional feature dumps:  
  `feats_val.npy`, `feats_test.npy`

### Stage-1+

- Prototype ensembles:  
  `prototype_ensemble_<criterion>.pt`
- Distance scale parameters:  
  `dist_scales_<criterion>.pt`
- Precomputed embeddings:  
  Auto-saved to the directory specified by `--ckpt_dir`
- Logs and visualizations (if W&B is enabled)

> These instructions match the setup used to produce results in our ICML 2025 paper. For additional visual analysis and
> ablation studies, refer to the [Notebooks](#notebooks) section.


---

# Results

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
@article{to2025diverse,
  title={Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift},
  author={To, Minh Nguyen Nhat and RWilson, Paul F and Nguyen, Viet and Harmanani, Mohamed and Cooper, Michael and Fooladgar, Fahimeh and Abolmaesumi, Purang and Mousavi, Parvin and Krishnan, Rahul G},
  journal={arXiv preprint arXiv:2505.23027},
  year={2025}
}
```

Acknowledgements
----------------

Some of the training and evaluation infrastructure in this repository was adapted
from:

- https://github.com/YyzHarry/SubpopBench

- https://github.com/dlmacedo/entropic-out-of-distribution-detection

We thank the authors for releasing their well-organized benchmark and codebase.



<p align="left">
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-red.svg?logo=github" alt="License: MIT">
  </a>
  <a href="https://github.com/minhto2802/dpe4subpop/stargazers">
    <img src="https://img.shields.io/github/stars/minhto2802/dpe4subpop.svg?style=flat&logo=github" alt="GitHub stars">
  </a>
  <a href="https://github.com/minhto2802/dpe4subpop/network">
    <img src="https://img.shields.io/github/forks/minhto2802/dpe4subpop.svg?style=flat&logo=github" alt="GitHub forks">
  </a>
  <a href="https://shields-visitor-count.onrender.com">
    <img src="https://visitor-badge.laobi.icu/badge?page_id=minhto2802.dpe4subpop&style=for-the-badge" alt="Visitors">
  </a>
</p>
