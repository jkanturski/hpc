#!/bin/bash
#BSUB -J setfit_train
#BSUB -q night
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=64GB]"
#BSUB -o setfit_%J.out
#BSUB -e setfit_%J.err

# --- environment setup ---
source ~/miniforge3/etc/profile.d/conda.sh
conda activate api_llm

# --- thread control (match allocated cores) ---
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32
export TORCH_NUM_THREADS=32

# Optional but often helpful on Power systems
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# --- run training ---
python t1.py
