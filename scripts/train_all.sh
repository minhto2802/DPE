#!/bin/bash

sbatch ./scripts/train.sh \
      --dataset_name Waterbirds \
      --model_name resnet152 \
      --epochs 400  # 400 113 # resnext101 resnet152

sbatch ./scripts/train.sh \
      --dataset_name CelebA \
      --epochs 50  # 50 20

#sbatch ./scripts/train.sh \
#      --dataset_name MetaShift \
#      --epochs 100  # 100 237
#

#sbatch ./scripts/train.sh \
#      --dataset-name ImagenetBG \
#      --epochs -1 \
#      --batch-size 108 \
#      --lr 1.e-3 \
#      --no-augmentation

#sbatch ./scripts/train.sh \
#      --dataset-name NICOpp \
#      --epochs -1 \
#      --batch-size 108 \
#      --lr 1.e-3 \
#      --no-augmentation

#sbatch ./scripts/train.sh \
#      --dataset_name Living17 \
#      --epochs 50  # 50  30
#
#sbatch ./scripts/train.sh \
#      --dataset_name CheXpertNoFinding \
#      --epochs 30  # 30  13

#sbatch ./scripts/train.sh \
#      --dataset-name CivilCommentsFine \
#      --epochs -1 \
#      --batch-size 32 \
#      --lr 1.e-5 \
#      --optim bert_adam \
#      --no-augmentation

#sbatch ./scripts/train.sh \
#      --dataset_name MultiNLI \
#      --epochs 4 \
#      --batch_size 32 \
#      --lr 1.e-5 \
#      --optim bert_adam \
#      --no_augmentation