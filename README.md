# Diverse Prototypical Ensembles (DPE)

This repository provides PyTorch-based scripts and SLURM job files for training and evaluating **Diverse Prototypical Ensembles (DPE)** on subpopulation shift benchmarks such as **Waterbirds**, **CelebA**, and **MultiNLI**.

---

## 🔧 Project Structure

- `main.py`: Main training script (handles both ERM baseline and prototypical ensemble stages).
- `scripts/train.sh`: Runs stage-0 (ERM) training.
- `scripts/train_pe.sh`: Runs stage-1+ (prototypical ensemble) training.
- `scripts/train_all.sh`: Runs all ERM jobs for multiple datasets.
- `scripts/train_all_pe.sh`: Runs all ensemble jobs for multiple datasets.
- `logs/`: Contains output logs (`%x.%j.log`).

---

## 🚀 Quick Start

### 1. Stage 0: Train ERM Baseline

```bash
sbatch scripts/train.sh \
    --dataset_name Waterbirds \
    --model_name resnet152 \
    --epochs 400
