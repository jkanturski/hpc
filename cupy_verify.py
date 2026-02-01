import cupy as cp
import numpy as np

# Create an array on the CPU
x_cpu = np.array([1, 2, 3])

# Transfer it to the V100 GPU
x_gpu = cp.asarray(x_cpu)

# Perform math on the GPU
y_gpu = x_gpu * 2

print(f"Result from GPU: {y_gpu}")
print(f"Current Device: {cp.cuda.Device()}")
