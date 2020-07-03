#!/bin/bash 
 
#SBATCH --export=ALL 
#SBATCH --output=predict.out 
 
#SBATCH --partition=inspur-gpu-ib 
#SBATCH --gres=gpu:1 
 
 
source /home/blao/rgz_rcnn/bashrc 
 
python /home/blao/pulsar_machine_learning/classify_pulsars_resnet/classify_pulsars_resnet/resnet/predict_pulsars.py
