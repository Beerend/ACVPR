#!/bin/sh
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH -c 32
python3 resnet.py --testing