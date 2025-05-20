Overview
========

This repository contains code to reproduce results from the ICML 2025 paper:

**Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift**  
Summary site: https://minhto2802.github.io/diversified_prototypical_ensemble

We present a simple yet effective method that improves robustness to subpopulation shifts without requiring group annotations. Our approach combines a pretrained backbone with a *diversified ensemble of prototype classifiers* trained to capture different substructure in the data.

The pipeline includes:

- Stage-0: Supervised backbone pretraining (ERM or IsoMax)
- Stage-1+: Training multiple diversified prototype heads

---

Directory Structure
-------------------

```
.
├── main.py                    # Unified entry point (stage-0 and stage-1+)
├── models/                    # Backbone & head architectures
├── utils/                     # Metrics, evaluation, feature extraction
├── scripts/
│   ├── train.sh               # Stage-0: ERM/IsoMax job
│   ├── train_all.sh           # Stage-0: all datasets
│   ├── train_pe.sh            # Stage-1+: ensemble training
│   └── train_all_pe.sh        # Stage-1+: all datasets
└── logs/                      # SLURM job outputs
```

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

Code Overview
----------------

- `main.py` handles both stages
  - `stage=0`: trains backbone from scratch
  - `stage>0`: adds diversified prototype heads (frozen backbone)

Key Functions:
- `train_model(...)`: core training loop
- `evaluate_phase(...)`: computes val/test metrics + logs
- `evaluate_ensemble_fixed_backbone(...)`: averages logits from heads
- `extract_features(...)`, `get_pre_extracted_features(...)`: numpy feature interface
---

Key Arguments
----------------

**General**
- `--dataset_name`: Waterbirds, CelebA, MultiNLI, etc.
- `--model_name`: resnet50, resnet152, bert-base-uncased
- `--epochs`, `--lr`: standard optimizer config
- `--seed`: random seed

**Stage-0**
- `--loss_name`: `ce` or `isomax`
- `--train_mode`: `full` (default)

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
- On W&B, the metrics of interest are in the sections with the prefix `ensemble_` (e.g. `ensemble_worst_group_acc` section)
- Tune `--cov_reg` (e.g. 1e4–1e6) to control prototype diversity.
- For IsoMax: `--entropic_scale` range varies between 10 to 40 depending on the datasets.
- `--subsample_type group` when `--train_attribute yes` will do subgroup balanced subsampling, while `--train_attributes no` will do class balanced subsampling on the training set (Stage-0) or the validation set (Stage-1+). 
- Stage-1+ training typically requires 15–30 epochs.
- Checkpoints: `/checkpoint/$USER/$SLURM_JOB_ID/ckpt_*.pt`
- Logs:        `logs/<jobname>.<id>.log`
- W&B group:   Controlled via `--wdb_group`
- Set `--no_wandb` flag to disable W&B logging (useful for debugging)
---

Expected Files
-----------------

**Stage-0 Outputs**
- `ckpt_best_acc.pt`, `ckpt_best_bal_acc.pt`, `ckpt_last.pt`
- `feats_val.npy`, `feats_test.npy` (optional feature dumps)

**Stage-1+ Outputs**
- `prototype_ensemble_<criterion>.pt`
- `dist_scales_<criterion>.pt`
- W&B run logs (optional)
---

Results
-------------------------

Worst-group accuracy on datasets **without subgroup annotations**:

| Algorithm        | Waterbirds | CelebA | CivilComments | MultiNLI | MetaShift | CheXpert | ImageNetBG | NICO++ | Living17 |
|------------------|------------|--------|----------------|-----------|------------|-----------|-------------|--------|-----------|
| ERM*             | 77.9±3.0   | 66.5±2.6 | 69.4±1.2       | 66.5±0.7  | 80.0±0.0   | 75.6±0.4  | 86.4±0.8    | 33.3±0.0 | 53.3±0.9  |
| RWY              | 86.1±0.7   | 82.9±2.2 | 67.5±0.6       | 68.0±1.9  | -          | -         | -           | -       | -         |
| AFR              | 90.4±1.1   | 82.0±0.5 | 68.7±0.6       | 73.4±0.6  | -          | -         | -           | -       | -         |
| ERM* + DPE (Ours)| 94.1±0.2   | 84.6±0.8 | 68.9±0.6       | 70.9±0.8  | 83.6±0.9   | 76.8±0.1  | 88.1±0.7    | 50.0±0.0 | 63.0±1.7  |



Worst-group accuracy on datasets **with subgroup annotation**:

| Algorithm            | Group Info<br>(Train / Val) | WATERBIRDS | CELEBA    | CIVILCOMMENTS | MULTINLI | METASHIFT | CHEXPERT  |
|----------------------|---------------------------|------------|-----------|----------------|-----------|------------|-----------|
| ERM*                 | X / X                     | 77.9±3.0   | 66.5±2.6  | 69.4±1.2       | 66.5±0.7  | 80.0±0.0   | 75.6±0.4  |
| Group DRO            | ✓ / ✓                     | 91.4±1.1   | 88.9±2.3  | 70.0±2.0       | 77.7±1.4  | -          | -         |
| RWG                  | ✓ / ✓                     | 87.6±1.6   | 84.3±1.8  | 72.0±1.9       | 69.6±1.0  | -          | -         |
| JTT                  | X / ✓                     | 86.7       | 81.1      | 69.3           | 72.6      | -          | -         |
| CnC                  | X / ✓                     | 88.5±0.3   | 88.8±0.9  | 68.9±2.1       | -         | -          | -         |
| SSA                  | X / ✓✓                    | 89.0±0.6   | 89.8±1.3  | 69.9±2.0       | 76.6±0.7  | -          | -         |
| DFR*                 | X / ✓✓                    | 92.9±0.2   | 88.3±1.1  | 70.1±0.8       | 74.7±0.7  | -          | -         |
| GAP (Last Layer)     | X / ✓✓                    | 93.2±0.2   | 90.2±0.3  | -              | 74.3±0.2  | -          | -         |
| GAP (All Layer)      | X / ✓✓                    | 93.8±0.1   | 90.2±0.3  | -              | 77.8±0.6  | -          | -         |
| ERM* + DPE (ours)    | X / ✓✓                    | 94.1±0.4   | 90.3±0.7  | 70.8±0.8       | 75.3±0.5  | 91.7±1.3   | 76.0±0.3  |

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
