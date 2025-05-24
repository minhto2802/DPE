#!/bin/bash

#sbatch ./scripts/train.sh \
#      --dataset_name Waterbirds \
#      --model_name resnet50 \
#      --epochs 400 \
#      --pretrained_imgnet

#sbatch ./scripts/train.sh \
#      --dataset_name CelebA \
#      --epochs 50  # 50 20
#      --pretrained_imgnet \

#sbatch ./scripts/train.sh \
#      --dataset_name MetaShift \
#      --epochs 100  # 100 237
#      --pretrained_imgnet \


#sbatch ./scripts/train.sh \
#      --dataset_name ImagenetBG \
#      --epochs 20 \
#      --batch_size 108 \
#      --lr 1.e-3 \
#      --no_augmentation

#sbatch ./scripts/train.sh \
#      --dataset_name NICOpp \
#      --epochs -1 \
#      --batch_size 108 \
#      --lr 1.e-3 \
#      --no_augmentation

sbatch ./scripts/train.sh \
      --dataset_name Living17 \
      --epochs 100
#      --no_augmentation
      # --epochs 50

#sbatch ./scripts/train.sh \
#      --dataset_name CheXpertNoFinding \
#      --epochs 30  # 30  13

#sbatch ./scripts/train.sh \
#      --dataset_name CivilCommentsFine \
#      --epochs -1 \
#      --batch_size 32 \
#      --lr 1.e-5 \
#      --optim bert_adam \
#      --no_augmentation

#sbatch ./scripts/train.sh \
#      --dataset_name MultiNLI \
#      --epochs 4 \
#      --batch_size 32 \
#      --lr 1.e-5 \
#      --optim bert_adam \
#      --no_augmentation