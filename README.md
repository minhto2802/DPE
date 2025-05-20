README
======

Robust Models under Sub-Population Shift
----------------------------------------

This repository accompanies the paper **“Diverse Prototypical Ensembles Improve Robustness to Subpopulation Shift”** (ICML 2025).
It contains two training stages:

1. **Stage-0 – Supervised ERM / IsoMax pre-training**  
   - Script: `scripts/train.sh`  
   - Launcher: `scripts/train_all.sh`

2. **Stage-1 – Diversified Prototype Ensemble (DPE)**  
   - Script: `scripts/train_pe.sh`  
   - Launcher: `scripts/train_all_pe.sh`

The Stage-0 backbone is frozen (or optionally fine-tuned) and multiple
prototype heads are appended. Each head is trained on a balanced subset of the
validation data while an inter-prototype similarity penalty encourages diverse
decision boundaries; logits are averaged at inference.


Quick start
-----------

    # Stage-0  (example on Waterbirds)
    sbatch scripts/train.sh --dataset_name Waterbirds --model_name resnet152 --epochs 400

    # Stage-1  (uses the ckpt produced above)
    sbatch scripts/train_pe.sh --dataset_name Waterbirds \
        --pretrained_path /scratch/.../Waterbirds/*/ckpt_last.pt \
        --epochs 20 --lr 1e-3 --cov_reg 1e5

To reproduce the paper numbers, launch the provided “*_all*.sh” meta-scripts on
three seeds (0/1/2). Logs are written to `logs/`, checkpoints to
`/checkpoint/$USER/$SLURM_JOB_ID/`, and run metrics to **Weights & Biases**.


Dependencies
------------

* Python ≥ 3.9
* PyTorch ≥ 1.13 + CUDA 11.x
* torchvision, numpy, pandas, seaborn, scikit-learn
* wandb (optional)
* Slurm scheduler with CUDA GPUs (tested on A40/RTX6000)

To create an environment:

    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt     # create this file if missing


Project layout
--------------

    .
    ├── main.py            # Stage-1 training (prototype ensemble)
    ├── main_v1.py         # Stage-0 training
    ├── models/            # backbones + heads
    ├── utils/             # timer, metrics, plotting
    ├── scripts/
    │   ├── train.sh       # stage-0 one run
    │   ├── train_all.sh
    │   ├── train_pe.sh    # stage-1 one run
    │   └── train_all_pe.sh
    └── logs/              # Slurm outputs


Key code
--------

- `train_model`   – unified loop (supports stage 0/1, freeze/full)
- `evaluate_phase` – isolated validation / test logic
- `evaluate_ensemble_fixed_backbone` – averages logits of all heads
- `eval_model`    – feature extraction helper
- `extract_features`, `get_pre_extracted_features` – offline numpy dumps


Signals & resumption
--------------------

Jobs are submitted with `--signal=SIGUSR1@90`; the handler in `main.py`
checkpoints before pre-emption. Resubmit with `--pretrained_path
/path/to/ckpt_last.pt` to resume.


Troubleshooting
---------------

- **CUDA OOM** → lower `--batch_size` or increase gradient accumulation.
- **wandb offline** → `export WANDB_MODE=offline`.
- **Dataset path** → set `DATA_DIR` env or `--data_dir` flag.


Citation
--------

    @inproceedings{to2025prototypicalensemble,
      title     = {Improving Robustness to Subpopulation Shifts by Heuristic Subspace Exploration with Enhanced Diversification},
      author    = {Minh To and collaborators},
      booktitle = {International Conference on Machine Learning},
      year      = {2025}
    }
