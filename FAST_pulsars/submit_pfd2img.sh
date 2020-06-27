#!/bin/bash 


#SBATCH --partition=purley-cpu
#SBATCH --time=12:00:00
#SBATCH --job-name=pfd2img
#SBATCH --nodes=1
#SBATCH --mem=60gb
#SBATCH --output=pfd.out
#SBATCH --export=all
#SBATCH --nodelist=purley-x86-cpu[03]

module use /home/app/modulefiles
module load presto/cpu-master

python /home/blao/pulsar_machine_learning/classify_pulsars_resnet/classify_pulsars_resnet/FAST_pulsars/pfd_to_img.py
