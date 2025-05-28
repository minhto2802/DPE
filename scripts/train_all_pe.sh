#!/bin/bash

CKPT_NAME=ckpt_last

S_DIR=/scratch/ssd004/scratch/minht/checkpoints/sd2
seed=0

sbatch ./scripts/train_pe.sh \
      --dataset_name Waterbirds \
      --pretrained_path $S_DIR/Waterbirds/*/${CKPT_NAME}.pt \
      --epochs 20 \
      --batch_size 256 \
      --lr 1.e-3 \
      --cov_reg 5.e5 \
      --seed ${seed} \
      -es 30 \
      $@

#sbatch ./scripts/train_pe.sh \
#      --dataset_name CelebA \
#      --pretrained_path $S_DIR/CelebA/*/${CKPT_NAME}.pt \
#      --epochs 20 \
#      --batch_size 256 \
#      --lr 5.e-4 \
#      --cov_reg 5.e5 \
#      --seed ${seed} \
#      -es 30 \
#      $@
#
#sbatch ./scripts/train_pe.sh \
#      --dataset_name MetaShift \
#      --pretrained_path $S_DIR/MetaShift/*/${CKPT_NAME}.pt \
#      --epochs 20 \
#      --cov_reg 1.e5 \
#      --batch_size 64 \
#      --optim sgd \
#      --seed ${seed} \
#      -es 20 \
#      $@
#sbatch ./scripts/train_pe.sh \
#      --dataset_name CheXpertNoFinding \
#      --pretrained_path $S_DIR/CheXpertNoFinding/*/${CKPT_NAME}.pt \
#      --epochs 20 \
#      --batch_size 256 \
#      --cov_reg 5.e5 \
#      --wd_weight 10 \
#      --lr 1.e-3 \
#      -es 20 \
#      --seed ${seed} \
#      $@
#sbatch ./scripts/train_pe.sh \
#      --dataset_name ImagenetBG \
#      --pretrained_path $S_DIR/ImageNetBG/*/${CKPT_NAME}.pt \
#      --epochs 20 \
#      --cov_reg 1.e5 \
#      --batch_size 256 \
#      --lr 1.e-3 \
#      -ec last \
#      -es 10 \
#      --seed ${seed} \
#      $@
#sbatch ./scripts/train_pe.sh \
#      --dataset_name NICOpp \
#      --pretrained_path $S_DIR/NICOpp/*/${CKPT_NAME}.pt \
#      --epochs 20 \
#      --cov_reg 1.e5 \
#      --batch_size 256 \
#      -es 30 \
#      --seed ${seed} \
#      $@
#sbatch ./scripts/train_pe.sh \
#      --dataset_name Living17 \
#      --pretrained_path $S_DIR/Living17/*/${CKPT_NAME}.pt \
#      --epochs 30 \
#      --cov_reg 1.e5 \
#      --batch_size 256 \
#      --lr 5.e-5 \
#      -ec last \
#      -es 10 \
#      --seed ${seed} \
#      $@
#
#sbatch ./scripts/train_pe.sh \
#      --dataset_name MultiNLI \
#      --epochs 20 \
#      --batch_size 256 \
#      --pretrained_path $S_DIR/MultiNLI/*/${CKPT_NAME}.pt \
#      --cov_reg 5.e6 \
#      --wd_weight 10 \
#      --lr 1.e-4 \
#      -es 30 \
#      --seed ${seed} \
#      $@
#sbatch ./scripts/train_pe.sh \
#      --dataset_name CivilCommentsFine \
#      --pretrained_path $S_DIR/CivilCommentsFine/*/${CKPT_NAME}.pt \
#      --epochs 20 \
#      --batch_size 256 \
#      --cov_reg 1.e5 \
#      --wd_weight 10 \
#      --lr 1.e-4 \
#      -es 30 \
#      --seed 0 \
#      $@ # 13620512
#
#sbatch ./scripts/train_pe.sh \
#      --dataset_name MultiNLI \
#      --epochs 20 \
#      --batch_size 256 \
#      --pretrained_path $S_DIR/MultiNLI/*/${CKPT_NAME}.pt \
#      --cov_reg 5.e6 \
#      --wd_weight 10 \
#      --lr 1.e-4 \
#      -es 30 \
#      --seed ${seed} \
#      $@
#sbatch ./scripts/train_pe.sh \
#      --dataset_name CivilCommentsFine \
#      --pretrained_path $S_DIR/CivilCommentsFine/*/${CKPT_NAME}.pt \
#      --epochs 20 \
#      --batch_size 256 \
#      --cov_reg 1.e5 \
#      --wd_weight 10 \
#      --lr 1.e-4 \
#      -es 30 \
#      --seed 0 \
#      $@
