#!/bin/bash
#BSUB -J torch_cpu
#BSUB -q short
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -W 00:30
#BSUB -o torch_%J.out
#BSUB -e torch_%J.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate api_llm

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

python t2.py
