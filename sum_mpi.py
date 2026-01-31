import socket
from mpi4py import MPI

host = socket.gethostname()
print(f"Host: {host}")

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

local_value = rank + 1
total = comm.reduce(local_value, op=MPI.SUM, root=0)

if rank == 0:
    print("Sum of ranks:", total)
