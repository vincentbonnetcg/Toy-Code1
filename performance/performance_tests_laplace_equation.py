"""
@author: Vincent Bonnet
@description : Solve Laplace's Equation on CPU and GPU using Jacobi Method
    Jacobi method solves Ax=b where
    A contains the coefficient of the discrete laplace operator
    0 -1  0
    -1 4 -1
    0 -1  0
    x is the unknown discretized function (array)
    b is equal to zero
    By definition, The jacobi iteration is :
    xi(k+1) = 1/aii * (bi - sum(aij * xj(k)) 'where j!=i')
    because b is a zero array and aii reference the coefficient 4
    xi(k+1) = 1/4 * (- sum(aij * xj(k)) 'for j!=i')
    and aij are -1 for j!=i
    => xi(k+1) = 1/4 * (sum(xj(k)) 'for j!=i')
"""

import time
import math
import numpy as np
from numba import njit, cuda, vectorize, prange, float32
import matplotlib.pyplot as plt
import skimage

'''
 Global Parameters
'''
NUM_NODES = 128 # For GPU should be a multiplier of TPB
JACOBI_ITERATIONS = 10000
TPB = 16 # Thread per block
GPU_SHARED_MEMORY_SIZE = TPB + 2 # +2 because the stencil requires -1 and +1

# TODO - Test Image restoration using partial differential equations
# TODO - improve shared memory - http://jinjiren.github.io/blog/cuda-programming/

def create_domain_from_image(num_nodes=64):
    '''
    Returns domain nodes along x, y and grid values
    The values come from an image
    '''
    # Convert image to greyscale numpy 2D-array
    image = skimage.img_as_float(skimage.color.rgb2gray(skimage.data.chelsea())).astype(np.float32)
    # Resize the image to (num_nodes, num_nodes) shape
    image = skimage.transform.resize(image, output_shape=(num_nodes, num_nodes), anti_aliasing=True)
    # Flip the image upside-down
    image = np.flipud(image)
    # Finally, turn the image into a single memory block to be used by Cuda
    image = np.ascontiguousarray(image, dtype=np.float32)

    return image

def create_domain(num_nodes=64):
    '''
    Returns domain nodes along x, y and grid values
    '''
    values = np.zeros((num_nodes, num_nodes), np.float) # unknown function
    values[num_nodes-1:num_nodes] = 1000.0

    return values

'''
 CPU Methods (Python and Numba)
'''
def jacobi_solver(x, next_x):
    num_dof_x = x.shape[0]
    num_dof_y = x.shape[1]

    for i in range(1, num_dof_x - 1):
        for j in range(1, num_dof_y - 1):
            next_x[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1]) * 0.25

@njit(parallel=True)
def numba_jacobi_solver(x, next_x):
    num_dof_x = x.shape[0]
    num_dof_y = x.shape[1]

    for i in prange(1, num_dof_x - 1):
        for j in range(1, num_dof_y - 1):
            next_x[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1]) * 0.25

'''
 CUDA Methods (Global and Shared Memory)
'''
@cuda.jit
def cuda_kernel_jacobi_solver(x, next_x):
    #i = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    #j = cuda.threadIdx.y + (cuda.blockIdx.y * cuda.blockDim.y)
    i, j = cuda.grid(2)
    if i > 0 and i < x.shape[0]-1 and j > 0 and j < x.shape[1]-1:
        next_x[i, j] = (x[i-1, j] + x[i+1, j] + x[i, j-1] + x[i, j+1]) * 0.25

@cuda.jit
def cuda_kernel_jacobi_solver_with_shared_memory(x, next_x):
    #i = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    #j = cuda.threadIdx.y + (cuda.blockIdx.y * cuda.blockDim.y)
    i, j = cuda.grid(2)
    if i <= 0 or i >= x.shape[0]-1 or j <= 0 or j >= x.shape[1]-1:
        return

    # Preload data into shared memory
    # TODO - make sure all threads are involved in the data preloading and no duplicate exists
    shared_x = cuda.shared.array(shape=(GPU_SHARED_MEMORY_SIZE, GPU_SHARED_MEMORY_SIZE), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    for idx in range(-1, 2):
        for idy in range(-1, 2):
            shared_x[idx+tx+1, idy+ty+1] = x[i+idx, j+idy]

    cuda.syncthreads()  # Wait for all threads to finish

    # Resources
    si = cuda.threadIdx.x + 1
    sj = cuda.threadIdx.y + 1
    result = (shared_x[si-1, sj] + shared_x[si+1, sj] + shared_x[si, sj-1] + shared_x[si, sj+1]) * 0.25
    next_x[i, j] = result

'''
 Dispatcher methods to call GPU or CPU methods
'''
def dispatch_gpu_algo(x, algo = cuda_kernel_jacobi_solver, num_iterations = 500):
    # Compute blocks
    threadsPerBlock = (TPB, TPB)
    blocksPerGridX = math.ceil(x.shape[0] / threadsPerBlock[0])
    blocksPerGridY = math.ceil(x.shape[1] / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGridX, blocksPerGridY)

    # Create array on gpu
    device_array = cuda.to_device(x)
    device_next_array = cuda.to_device(np.copy(x))
    buffer_id = 0
    buffers = [device_array, device_next_array]

    # Run kernel
    start_time = time.time()
    for iteration in range(num_iterations):
        algo[blocksPerGrid, threadsPerBlock](buffers[buffer_id], buffers[(buffer_id+1)%2])
        buffer_id = (buffer_id+1)%2

    end_time = time.time()
    log = 'cuda kernels - %f sec' % (end_time - start_time)
    print(log)

    # copy gpu data back to
    device_array.copy_to_host(x)

def dispatch_cpu_algo(x, algo = jacobi_solver, num_iterations = 500):
    next_x = np.copy(x)
    buffer_id = 0
    buffers = [x, next_x]
    for iteration in range(num_iterations):
        algo(buffers[buffer_id], buffers[(buffer_id+1)%2])
        buffer_id = (buffer_id+1)%2

def dummy_call_to_compile_jit_function():
    '''
    Dummy - only for proper profiling
    '''
    values = create_domain(2)
    dispatch_cpu_algo(values, numba_jacobi_solver, 1)

def main():
    '''
    Main
    '''
    #. pre-compile cpu jit function so timing doesn't
    dummy_call_to_compile_jit_function()

    #1. Create 2D grid domain
    values = create_domain(NUM_NODES)
    #values = create_domain_from_image(NUM_NODES)

    #2. Run and benchmark algorithm
    start_time = time.time()
    #dispatch_cpu_algo(values, jacobi_solver, JACOBI_ITERATIONS)
    dispatch_cpu_algo(values, numba_jacobi_solver, JACOBI_ITERATIONS)
    #dispatch_gpu_algo(values, cuda_kernel_jacobi_solver, JACOBI_ITERATIONS)
    #dispatch_gpu_algo(values, cuda_kernel_jacobi_solver_with_shared_memory, JACOBI_ITERATIONS)
    end_time = time.time()
    print('Timing - %f sec' % (end_time - start_time))

    #3. Show result
    fig, ax = plt.subplots()
    domain = np.linspace(0, 10, num=NUM_NODES, endpoint=True)
    im = ax.pcolormesh(domain, domain, values, cmap="rainbow", antialiased=True, shading="gouraud")
    fig.colorbar(im)
    #fig.savefig("test.png")

if __name__ == '__main__':
    main()

