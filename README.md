# <u>D</u>iverse <u>P</u>rototypical <u>E</u>nsembles Improve Robustness to Subpopulation Shift

![Diverse Prototypical Ensemble Training Pipeline](docs/figures/embeddings_figure.png)

---

Overview
========

This repository contains code to reproduce results from the ICML 2025 paper.  
Summary site: https://minhto2802.github.io/diversified_prototypical_ensemble

We present a simple yet effective method that improves robustness to subpopulation shifts without requiring group
annotations. Our approach combines a pretrained backbone with a *diversified ensemble of prototype classifiers* trained
to capture different substructure in the data.

The pipeline includes:

- Stage-0: Supervised backbone pretraining (ERM or IsoMax)
- Stage-1+: Training multiple diversified prototype heads

---

Notebooks
---------

We provide several Jupyter notebooks under the [`notebooks/`](notebooks/) directory for qualitative analysis, controlled experiments, and ablation studies:

- **[`00_synthetic.ipynb`](notebooks/00_synthetic.ipynb)**  
  Demonstrates a 2D synthetic example simulating subpopulation shift.  
  Includes visualization of individual subgroup-specific classifiers, prototype-based decision boundaries, and the benefit of ensemble aggregation.

- **[`01_waterbirds_with_attribute_annotation.ipynb`](notebooks/01_waterbirds_with_attribute_annotation.ipynb)**  
  End-to-end run of Diverse Prototypical Ensembles on the Waterbirds dataset using validation subgroup labels.  
  Includes per-group accuracy plots and analysis of representation coverage.

- **[`02_celeba_without_attribute_annotation.ipynb`](notebooks/02_celeba_without_attribute_annotation.ipynb)**  
  Training and evaluation on CelebA under the more realistic setting where subgroup labels are not available.  
  This demonstrates DPE's ability to improve worst-group performance without explicit group supervision.

Each notebook is self-contained and can be run independently for demonstration or extension to other datasets.

---

Quickstart
----------

**Stage-0 training (ERM or IsoMax):**

```
sbatch scripts/train.sh \
    --dataset_name Waterbirds \
    --model_name resnet152 \
    --epochs 400
```

**Stage-1+ training (Diversified Prototypes):**

```
sbatch scripts/train_pe.sh \
    --dataset_name Waterbirds \
    --pretrained_path /scratch/.../Waterbirds/*/ckpt_last.pt \
    --epochs 20 \
    --lr 1e-3 \
    --cov_reg 1e5
```

Launch all jobs:

```
sbatch scripts/train_all.sh
sbatch scripts/train_all_pe.sh
```

Key Arguments
----------------

**General**

- `--dataset_name`: Waterbirds, CelebA, MultiNLI, etc.
- `--model_name`: resnet50, bert-base-uncased
- `--epochs`, `--lr`: standard optimizer config
- `--seed`: random seed

**Stage-0**

- `--loss_name`: `ce` (default) or `isomax`
- `--train_mode`: `full` (default) or `freeze`

**Stage-1+**

- `--stage 1`
- `--pretrained_path`: path to ckpt from stage 0
- `--num_stage`: number of ensemble heads (default: 16)
- `--cov_reg`: strength of prototype decorrelation
- `--subsample_type`: `group` or `class` (data balancing)
- `--entropic_scale`: IsoMax hyperparam
- `--train_mode freeze`: linear-probe protocol
- `-ncbt`: disable class-balanced training
- `-sit`: shuffle training set each epoch

---

Training Tips
-----------------------

- On W&B, the metrics of interest are in the sections with the prefix `ensemble_` (e.g. `ensemble_worst_group_acc`
  section)
- Tune `--cov_reg` (e.g. 1e4–1e6) to control prototype diversity.
- For IsoMax: `--entropic_scale` range varies between 10 and 40 depending on the datasets.
- `--subsample_type group` when `--train_attribute yes` will do subgroup balanced subsampling, while
  `--train_attributes no` will do class balanced subsampling on the training set (Stage-0) or the validation set (
  Stage-1+).
- Stage-1+ training typically requires 15–30 epochs.
- Checkpoints: `/checkpoint/$USER/$SLURM_JOB_ID/ckpt_*.pt`
- Logs:        `logs/<jobname>.<id>.log`
- W&B group:   Controlled via `--wdb_group`
- Set `--no_wandb` flag to disable W&B logging (useful for debugging)

---

Expected Files
-----------------

**Stage-0 Outputs**

- `ckpt_best_acc.pt`, `ckpt_best_bal_acc.pt`, *`ckpt_last.pt`* (used in the paper)
- `feats_val.npy`, `feats_test.npy` (optional feature dumps)

**Stage-1+ Outputs**

- `prototype_ensemble_<wga_val>.pt` (with `wga_val` corresponding to the criterion for selecting the member of the
  prototypical ensemble)
- `dist_scales_<wga_val>.pt`
- W&B run logs (optional)
- Embeddings will be extracted and saved at the `--ckpt_dir` directory at the start of the DPE training (if not yet
  generated)

---

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

Much of the training and evaluation infrastructure in this repository was adapted from:

https://github.com/YyzHarry/SubpopBench  
We thank the authors for releasing their well-organized benchmark and codebase.
