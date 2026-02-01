from mpi4py import MPI
import cupy as cp
import socket

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

# Step 1: Map the rank to a local GPU ID
# Since V100 nodes usually have 4 GPUs, we use modulo 4
local_gpu_id = rank % 4 

try:
    cp.cuda.Device(local_gpu_id).use()
    
    # Step 2: Create a large array directly on the V100
    # A 1GB array (floats take 8 bytes, so 125 million elements)
    n = 125_000_000 
    data_gpu = cp.ones(n) * rank
    
    # Step 3: Do a calculation
    gpu_sum = data_gpu.sum()
    
    print(f"Rank {rank} on {hostname}: Using V100 ID {local_gpu_id}. Sum result: {gpu_sum}")

except Exception as e:
    print(f"Rank {rank} on {hostname} failed to access GPU: {e}")
