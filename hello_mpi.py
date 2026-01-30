from mpi4py import MPI
import socket

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
host = socket.gethostname()

print(f"Hello from rank {rank}/{size} on {host}")
