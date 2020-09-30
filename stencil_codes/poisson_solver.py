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

import math
import numba
import numpy as np
import matplotlib.pyplot as plt

NUM_NODES = 128 # For GPU should be a multiplier of TPB
JACOBI_ITERATIONS = 10000
TPB = 16 # Thread per block
GPU_SHARED_MEMORY_SIZE = TPB + 2 # +2 because the stencil requires -1 and +1

def create_domain(num_nodes=64):
    '''
    Returns domain nodes along x, y and grid values
    '''
    values = np.zeros((num_nodes, num_nodes), np.float) # unknown function
    values[num_nodes-1:num_nodes] = 1000.0

    return values

@numba.cuda.jit
def jacobi_kernel(x, next_x):
    # jacobi solver with an attempt of shared memory
    #i = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    #j = cuda.threadIdx.y + (cuda.blockIdx.y * cuda.blockDim.y)
    i, j = numba.cuda.grid(2)
    if i <= 0 or i >= x.shape[0]-1 or j <= 0 or j >= x.shape[1]-1:
        return

    # Preload data into shared memory
    # TODO - make sure all threads are involved in the data preloading and no duplicate exists
    shared_x = numba.cuda.shared.array(shape=(GPU_SHARED_MEMORY_SIZE, GPU_SHARED_MEMORY_SIZE), dtype=numba.float32)

    tx = numba.cuda.threadIdx.x
    ty = numba.cuda.threadIdx.y
    for idx in range(-1, 2):
        for idy in range(-1, 2):
            shared_x[idx+tx+1, idy+ty+1] = x[i+idx, j+idy]

    numba.cuda.syncthreads()  # Wait for all threads to finish

    # Resources
    si = numba.cuda.threadIdx.x + 1
    sj = numba.cuda.threadIdx.y + 1
    result = (shared_x[si-1, sj] + shared_x[si+1, sj] + shared_x[si, sj-1] + shared_x[si, sj+1]) * 0.25
    next_x[i, j] = result

def poisson_solver(domain, num_iterations):
    # Compute blocks
    threadsPerBlock = (TPB, TPB)
    blocksPerGridX = math.ceil(domain.shape[0] / threadsPerBlock[0])
    blocksPerGridY = math.ceil(domain.shape[1] / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGridX, blocksPerGridY)

    # Create array on GPU
    buffers = [None, None]
    buffers[0] = numba.cuda.to_device(domain)
    buffers[1] = numba.cuda.to_device(np.copy(domain))

    for iteration in range(num_iterations):
        jacobi_kernel[blocksPerGrid, threadsPerBlock](buffers[0], buffers[1])
        buffers[0], buffers[1] = buffers[1], buffers[0] # swap buffers

    # copy gpu data back to
    buffers[0].copy_to_host(domain)

    # show result
    fig, ax = plt.subplots()
    domain_points = np.linspace(0, 10, num=NUM_NODES, endpoint=True)
    im = ax.pcolormesh(domain_points, domain_points, buffers[0], cmap="rainbow", antialiased=True, shading="gouraud")
    fig.colorbar(im)
    #fig.savefig("test.png")


if __name__ == '__main__':
    domain = create_domain(NUM_NODES)
    poisson_solver(domain, JACOBI_ITERATIONS)

