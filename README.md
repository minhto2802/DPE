📄 Paper: *Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift*  
Website: [https://minhto2802.github.io/diversified_prototypical_ensemble](https://minhto2802.github.io/diversified_prototypical_ensemble)

Overview
---------------------
This repository contains code to reproduce the results from the ICML 2025 paper. It includes:

- Stage-0 supervised pretraining (ERM)
- Stage-1 ensemble training with diversified prototype heads

Training is SLURM-compatible. See scripts for submission templates. The pretrained backbone weights will be available for downloading shortly.

---

Directory Structure
---------------------

```
.
├── main.py                    # Training entry point
├── utils/                     # Utilities: metrics, models, etc.
├── scripts/
│   ├── train.sh               # Stage-0: single job
│   ├── train_all.sh           # Stage-0: all datasets
│   ├── train_pe.sh            # Stage-1+: single job
│   └── train_all_pe.sh        # Stage-1+: all datasets
└── logs/                      # SLURM logs
```

---

Quickstart
-------------

1. **Stage-0**: Supervised pretraining on target dataset

```
sbatch scripts/train.sh \
    --dataset_name Waterbirds \
    --model_name resnet50 \
    --epochs 400
```

2. **Stage-1+**: Ensemble training with frozen backbone

```
sbatch scripts/train_pe.sh \
    --dataset_name Waterbirds \
    --pretrained_path /scratch/.../Waterbirds/*/ckpt_last.pt \
    --epochs 20 \
    --lr 1e-3 \
    --cov_reg 1e5
```

To launch preconfigured jobs on multiple datasets:

```
sbatch scripts/train_all.sh
sbatch scripts/train_all_pe.sh
```

---

Code Overview
----------------

- `main.py`:
  - Unified entry point for both stage-0 and stage-1+.
  - Supports `train_mode: full` or `freeze`.
  - `stage=0`: trains classifier end-to-end.
  - `stage>0`: freezes backbone, trains multiple prototype heads on class-balanced or group-aware subsets.

- Core Functions:
  - `train_model(...)`: unified training loop
  - `evaluate_phase(...)`: logging + evaluation metrics
  - `evaluate_ensemble_fixed_backbone(...)`: inference with prototype averaging
  - `extract_features(...)`: pre-extract features from frozen backbone
  - `get_pre_extracted_features(...)`: manage numpy dumps and reuse

---

🧪 Reproducing Results
----------------------

1. Run Stage-0 (ERM or IsoMax) on seeds {0,1,2}:

```
sbatch scripts/train.sh --seed 0 ...
sbatch scripts/train.sh --seed 1 ...
sbatch scripts/train.sh --seed 2 ...
```

2. For each seed, update `S_DIR` in `train_all_pe.sh` to match saved checkpoints:

```
S_DIR=/scratch/ssd004/scratch/minht/checkpoints/sd${seed}
```

Then run:

```
bash scripts/train_all_pe.sh
```

3. Collect `worst-group accuracy` from `wandb` or logs.

---

⚙️ Arguments of Interest
------------------------

**Stage-0**
- `--loss_name ce|isomax`
- `--dataset_name`
- `--model_name`
- `--epochs`, `--lr`
- `--ckpt_dir`

**Stage-1+**
- `--pretrained_path` (required)
- `--stage 1`
- `--num_stage` (number of heads)
- `--train_mode freeze`
- `--subsample_type group|class`
- `--cov_reg` (inter-prototype penalty)
- `--entropic_scale` (for IsoMax)

---

Notes
--------

- Checkpoints: `/checkpoint/$USER/$SLURM_JOB_ID/ckpt_*.pt`
- Logs:        `logs/<jobname>.<id>.log`
- W&B group:   Controlled via `--wdb_group`

Use `--no_wandb` to disable logging.

---

Citation
-----------

```
@inproceedings{to2025dpe,
  title     = {Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift},
  author    = {To, Minh Nguyen Nhat and Wilson, Paul F R and Nguyen, Viet and Harmanani, Mohamed and Cooper, Michael and Fooladgar, Fahimeh and Abolmaesumi, Purang and Mousavi, Parvin and Krishnan, Rahul},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2025}
}
```
