from mpi4py import MPI
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Assign local GPU 0 or 1 based on rank
local_gpu_id = rank % 2 
cp.cuda.Device(local_gpu_id).use()

print(f"Rank {rank} is locked to GPU {local_gpu_id}")
