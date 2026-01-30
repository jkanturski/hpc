#!/bin/bash
#BSUB -J pytorch_cpu_train
#BSUB -q short
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -o pytorch_%J.out
#BSUB -e pytorch_%J.err

# --- environment setup ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate api_llm

# --- PyTorch CPU threading ---
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TORCH_NUM_THREADS=4

# --- run training ---
python t3.py
