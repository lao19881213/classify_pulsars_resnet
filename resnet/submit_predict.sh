#!/bin/bash 
 
#SBATCH --export=ALL 
#SBATCH --output=predict.out 
 
#SBATCH --partition=inspur-gpu-ib 
#SBATCH --gres=gpu:1 
 
 
source /home/blao/rgz_rcnn/bashrc 
 
python /home/blao/pulsar_machine_learning/tensorflow-cnn-finetune/resnet/predict_pulsars.py
