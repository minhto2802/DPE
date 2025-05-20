#!/bin/bash

#SBATCH -J pe
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --time 2:00:00
##SBATCH --qos=m4
##SBATCH --gres=gpu:rtx6000:1
##SBATCH --time 24:00:00
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --gres=gpu:a40:1
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90
##SBATCH --array=1-3
#SBATCH --exclude=gpu144

echo train-prototypes

export WANDB_RUN_ID=$SLURM_JOB_ID
S_DIR=/scratch/ssd004/scratch/minht/checkpoints/sd0
CKPT_NAME=ckpt_last

seed=$SLURM_ARRAY_TASK_ID
if [ -z "$seed" ]; then
  seed=0
fi

python main.py \
  --ckpt_dir /checkpoint/$USER/$SLURM_JOB_ID \
  --seed $seed \
  --epochs 20 \
  --loss_name isomax \
  --stage 1 \
  --pretrained_path $S_DIR/Waterbirds/*/${CKPT_NAME}.pt \
  --train_mode freeze \
  --num_stage 4 \
  --lr 1.e-3 \
  --train_attr yes \
  --dataset_name Waterbirds \
  --wdb_group dpe \
  --wd_weight 10 \
  --cov_reg 1.e5 \
  --subsample_type group \
  -ncbt \
  -sit \
  $@
