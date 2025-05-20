#!/bin/bash

#SBATCH -J stage0
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH -c 16
##SBATCH --time 12:00:00
##SBATCH --qos=m
##SBATCH --gres=gpu:rtx6000:1
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j.log
#SBATCH --time 24:00:00
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH --signal=SIGUSR1@90
#SBATCH --exclude=gpu134,gpu177,gpu138
#SBATCH --array=0,1,2
echo train-prototypes

export WANDB_RUN_ID=$SLURM_JOB_ID

seed=$SLURM_ARRAY_TASK_ID
if [ -z "$seed" ]; then
  seed=0 #
fi

python main_v1.py \
  --ckpt_dir /checkpoint/$USER/$SLURM_JOB_ID \
  --seed $seed \
  --epochs 20 \
  --loss_name ce \
  --dataset_name MultiNLI \
  --wdb_group supervised_v1k_scratch \
  $@