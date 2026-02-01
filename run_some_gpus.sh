#BSUB -J GPU_Part_1
#BSUB -q short
#BSUB -n 4                          # 4 ranks total
#BSUB -R "span[ptile=2]"            # 2 ranks per node
#BSUB -gpu "num=2:mode=exclusive_process:j_exclusive=yes"
#BSUB -o gpu_out_%J.out

module purge
module load compilers/nvidia/hpc_sdk/21.9
module load conda/opence/1.5.0

source ~/miniforge3/etc/profile.d/conda.sh
conda activate api_llm

# Spectrum MPI usually handles the GPU binding automatically on POWER9
mpirun python all_gpus.py
