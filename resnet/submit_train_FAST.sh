#!/bin/bash 
 
#SBATCH --export=ALL 
#SBATCH --output=train.out 
 
#SBATCH --partition=inspur-gpu-ib 
#SBATCH --gres=gpu:1 
 
 
source /home/blao/rgz_rcnn/bashrc 
 
python pulsars_trainval.py \
       --num_classes 2 \
       --num_epochs 10 \
       --train_layers "fc" \
       --batch_size 512 \
       --training_file "../data/FAST_train.txt" \
       --val_file  "../data/FAST_val.txt"
