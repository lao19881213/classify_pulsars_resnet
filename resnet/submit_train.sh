#!/bin/bash 
 
#SBATCH --export=ALL 
#SBATCH --output=train.out 
 
#SBATCH --partition=inspur-gpu-ib 
#SBATCH --gres=gpu:1 
 
 
source /home/blao/rgz_rcnn/bashrc 
 
python /home/blao/pulsar_machine_learning/tensorflow-cnn-finetune/resnet/pulsars_trainval.py \
       --num_classes 2 \
       --num_epochs 1 \
       --multi_scale "225,256" \
       --train_layers "fc,scale5,scale4/block6,scale4/block5"
