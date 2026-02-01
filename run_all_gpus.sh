#BSUB -J MPI_16_GPU
#BSUB -q short
#BSUB -n 16                 # 16 Total MPI ranks
#BSUB -R "span[ptile=2]"    # 2 ranks per node (matches 2 GPUs per node)
#BSUB -gpu "num=2:mode=exclusive_process" 
#BSUB -o gpu_out_%J.out

module purge
module load compilers/nvidia/hpc_sdk/21.9
module load conda/opence/1.5.0
conda activate api_llm

# Spectrum MPI usually handles the GPU binding automatically on POWER9
mpirun python all_gpus.py
