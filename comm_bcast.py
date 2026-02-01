import socket
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
host = socket.gethostname()

if rank == 0:
    # Only the leader (Rank 0) creates the data
    data = {'strategy': 'attack', 'coordinates': [42, 88]}
else:
    # Everyone else starts with nothing
    data = None

# The magic line: Rank 0 broadcasts 'data' to everyone else
data = comm.bcast(data, root=0)

print(f"Rank {rank} received data: {data} on: {host}")
