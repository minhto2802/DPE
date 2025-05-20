README
======

This repo contains two training pipelines:

1.  **Stage-0 supervised pre-training / fine-tuning**
    * main entry-point:  `main_v1.py`
    * batch script      :  `scripts/train.sh`
    * launcher          :  `scripts/train_all.sh`

2.  **Stage-1 diversified-prototype ensemble (DPE) training**
    * main entry-point:  `main.py`
    * batch script      :  `scripts/train_pe.sh`
    * launcher          :  `scripts/train_all_pe.sh`


──────────────────────────────────────────────────────────────────────────────
Prerequisites
──────────────
* Python ≥3.9 (tested with 3.10)
* PyTorch ≥1.13  + CUDA 11.x
* torchvision, numpy, pandas, seaborn, scikit-learn
* wandb    (for experiment tracking)
* Slurm    (A40/RTX nodes) with `sbatch`

Activate the environment (example):
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt            # create this if needed


──────────────────────────────────────────────────────────────────────────────
Folder structure
───────────────
.
├── main.py                     # Stage-1 (prototype ensemble)
├── main_v1.py                  # Stage-0 (ERM / IsoMax)
├── models/                     # model builders
├── utils/                      # timer, eval_metrics, etc.
├── scripts/
│   ├── train.sh                # single-stage ERM job
│   ├── train_all.sh            # submit ERM jobs for all datasets
│   ├── train_pe.sh             # single-stage DPE job
│   └── train_all_pe.sh         # submit DPE jobs for all datasets
└── logs/                       # Slurm stdout / stderr


──────────────────────────────────────────────────────────────────────────────
1.  Stage-0  —  Supervised ERM / IsoMax
─────────────
Edit `scripts/train.sh` (or pass `sbatch` overrides) to choose:

    --dataset_name          [Waterbirds | CelebA | MultiNLI | …]
    --model_name            resnet50 / resnet152 / bert-base-uncased …
    --epochs                (# training epochs)
    --loss_name             ce | isomax
    --wdb_group             wandb group tag

Submit one job:

    sbatch scripts/train.sh \
           --dataset_name Waterbirds \
           --model_name resnet152 \
           --epochs 400

Submit all preconfigured jobs:

    sbatch scripts/train_all.sh

Outputs
-------
* checkpoints:   `/checkpoint/${USER}/${SLURM_JOB_ID}/ckpt_*.pt`
* wandb run id:  "${SLURM_JOB_ID}"
* logs:          `logs/stage0.<job>.log`


──────────────────────────────────────────────────────────────────────────────
2.  Stage-1  —  Diversified Prototype Ensemble (DPE)
─────────────
Requires a pretrained backbone checkpoint from Stage-0.

Key arguments in `scripts/train_pe.sh`:

    --stage 1
    --pretrained_path  /path/to/ckpt_last.pt              # glob accepted
    --train_mode       freeze | full                      # “freeze” = linear-probe
    --num_stage        4                                  # number of prototype sets
    --cov_reg          1.e5                               # covariance penalty
    --subsample_type   group | class | None
    -ncbt              (flag) disable class balanced training
    -sit               (flag) shuffle indices at each epoch
    -es  <float>       entropic scale for IsoMax

Example (single job):

    sbatch scripts/train_pe.sh \
           --dataset_name Waterbirds \
           --pretrained_path /scratch/.../ckpt_last.pt \
           --lr 1e-3 --epochs 20 --cov_reg 5e5

Launch predefined set:

    sbatch scripts/train_all_pe.sh

Outputs
-------
* new ensemble checkpoints in `/checkpoint/${USER}/${SLURM_JOB_ID}/`
* wandb group “dpe”
* logs: `logs/pe.<job>.log`


──────────────────────────────────────────────────────────────────────────────
Python entry points
───────────────────
main_v1.py  (stage-0)
  ▸ Parses CLI arguments (`get_args()` in `args.py`).
  ▸ Builds dataset  → dataloaders  → model  → trainer (`train_model`).

main.py     (stage-1)
  ▸ Loads frozen backbone + prototype head(s).
  ▸ Iteratively calls `train_model` with `stage>0` to append new heads.
  ▸ `evaluate_ensemble_fixed_backbone` averages logits of all heads.

Key helpers
-----------
* `train_model`      – core training loop (supports freeze/full).
* `eval_model`       – feature / logits extraction for validation.
* `evaluate_phase`   – isolated validation/test logic.
* `extract_features` – dumps numpy feats for freezing runs.


──────────────────────────────────────────────────────────────────────────────
Reproducing paper numbers
────────────────────────
1.  Run `train_all.sh` on 3 different seeds (0/1/2) to create Stage-0
    checkpoints under `/scratch/.../checkpoints/sd<seed>/DATASET/...`.

2.  Update `train_all_pe.sh`
        S_DIR=/scratch/.../checkpoints/sd${seed}
    then launch for each seed:
        bash scripts/train_all_pe.sh

3.  Collect worst-group accuracy from wandb or terminal output.


──────────────────────────────────────────────────────────────────────────────
Signals & resubmission
─────────────────────
Slurm `--signal=SIGUSR1@90` traps 90 s before time-limit; the trap is
handled in `main.py` to checkpoint before job pre-emption.  To resume,
pass `--pretrained_path` to the latest `ckpt_last.pt` and re-submit.

──────────────────────────────────────────────────────────────────────────────
Troubleshooting
───────────────
* **OOM on A40** – lower `--batch_size`; increase `--gradient_accumulation_steps`.
* **Dataset path** – set `DATA_DIR` env or `--data_dir` arg.
* **wandb offline** – run `export WANDB_MODE=offline`.
* **CUDA mismatch** – `module load cuda/11.8` or edit shebang.

──────────────────────────────────────────────────────────────────────────────
Citation
────────
If you use this codebase, please cite:
    @inproceedings{to2025prototypicalensemble,
      title={Improving Robustness to Subpopulation Shifts by Heuristic Subspace Exploration with Enhanced Diversification},
      author={To, Minh and et al.},
      booktitle={ICML},
      year={2025}
    }
