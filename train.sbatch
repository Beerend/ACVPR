#!/bin/sh
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -c 32
python3 resnet.py --epochs 1 --save_dir './result_resnet_2'