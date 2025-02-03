import cupy as cp
import numpy as np
import time

# Free GPU memory before starting
cp.get_default_memory_pool().free_all_blocks()

# Matrix size (adjust if needed)
size = 20000  

# Create matrices directly on GPU with float32 (faster)
A = cp.random.rand(size, size, dtype=cp.float32)
B = cp.random.rand(size, size, dtype=cp.float32)

# Run matrix multiplication on GPU with CUDA Stream
start_time = time.time()
with cp.cuda.Stream():
    C = cp.dot(A, B)
cp.cuda.Device(0).synchronize()  # Ensure GPU finishes computation
gpu_time = time.time() - start_time

print(f"GPU time: {gpu_time:.5f} seconds")

# Convert to NumPy for CPU comparison
A_cpu = np.random.rand(size, size).astype(np.float32)
B_cpu = np.random.rand(size, size).astype(np.float32)

# Run matrix multiplication on CPU
start_time = time.time()
C_cpu = np.dot(A_cpu, B_cpu)
cpu_time = time.time() - start_time

print(f"GPU time: {gpu_time:.5f} seconds")
print(f"CPU time: {cpu_time:.5f} seconds")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")
