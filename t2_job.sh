#!/bin/bash
#BSUB -J cifar10_cpu
#BSUB -q short
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 01:00
#BSUB -o cifar10_%J.out
#BSUB -e cifar10_%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate api_llm

# Prevent oversubscription
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_NUM_THREADS=4

python t2.py
