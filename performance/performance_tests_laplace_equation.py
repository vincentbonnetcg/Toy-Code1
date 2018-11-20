"""
@author: Vincent Bonnet
@description : Evaluate CPU and stuff
"""

import time
import math
import numpy as np
from numba import njit, cuda, vectorize, prange
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
 Global Parameters
'''
NUM_NODES = 128
JACOBI_ITERATIONS = 1000

# add title to say which algo + resolution + iteration
# TODO - add time + graph
# TODO - try stencil
# TODO - Test Image restoration using partial differential equations

def create_domain(num_nodes=64):
    '''
    Returns domain nodes along x, y and grid values
    '''
    domain_x = np.linspace(0, 10, num=num_nodes, endpoint=True)
    domain_y = np.linspace(0, 10, num=num_nodes, endpoint=True)

    values = np.zeros((num_nodes, num_nodes)) # unknown function
    values[num_nodes-1:num_nodes] = 1000.0

    return domain_x, domain_y, values

def jacobi_solver(x, next_x,  num_iterations=500):
    '''
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
    '''
    num_dof_x = x.shape[0]
    num_dof_y = x.shape[1]

    for iteration in range(num_iterations):
        for i in range(1, num_dof_x - 1):
            for j in range(1, num_dof_y - 1):
                next_x[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1]) * 0.25

        np.copyto(x, next_x)

@njit(parallel=True)
def numba_jacobi_solver(x, next_x,  num_iterations = 500):
    num_dof_x = x.shape[0]
    num_dof_y = x.shape[1]

    for iteration in range(num_iterations):
        for i in prange(1, num_dof_x - 1):
            for j in range(1, num_dof_y - 1):
                next_x[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1]) * 0.25

        # numba doesn't support copyto
        #np.copyto(x, next_x)
        for i in prange(1, num_dof_x - 1):
            for j in range(1, num_dof_y - 1):
                x[i][j] = next_x[i][j]

@cuda.jit
def cuda_kernel_jacobi_solver(x, next_x):
    #i = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    #j = cuda.threadIdx.y + (cuda.blockIdx.y * cuda.blockDim.y)
    i, j = cuda.grid(2)
    if i > 0 and i < x.shape[0]-1 and j > 0 and j < x.shape[1]-1:
        next_x[i, j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1]) * 0.25

@cuda.jit
def cuda_kernel_copy(array_from, array_to):
    #i = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    #j = cuda.threadIdx.y + (cuda.blockIdx.y * cuda.blockDim.y)
    i, j = cuda.grid(2)
    if i > 0 and i < array_from.shape[0]-1 and j > 0 and j < array_from.shape[1]-1:
        array_to[i, j] = array_from[i, j]

def solve_laplace_equation_gpu(x, num_iterations = 500):
    # Compute blocks
    threadsPerBlock = (16, 16)
    blocksPerGridX = math.ceil(x.shape[0] / threadsPerBlock[0])
    blocksPerGridY = math.ceil(x.shape[1] / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGridX, blocksPerGridY)

    # Create array on gpu
    device_array = cuda.to_device(x)
    device_next_array = cuda.to_device(np.copy(x))

    # Run kernel
    start_time = time.time()
    for iteration in range(num_iterations):
        cuda_kernel_jacobi_solver[blocksPerGrid, threadsPerBlock](device_array, device_next_array)
        cuda_kernel_copy[blocksPerGrid, threadsPerBlock](device_next_array, device_array)
    end_time = time.time()
    log = 'cuda kernels - %f sec' % (end_time - start_time)
    print(log)

    # copy gpu data back to
    device_array.copy_to_host(x)

def solve_laplace_equation_cpu(x, algo = jacobi_solver, num_iterations = 500):
    next_x = np.copy(x)
    for iteration in range(0, num_iterations, 1):
        algo(x, next_x, 1)

# Dummy
def dummy_call_to_compile_jit():
    domain_x, domain_y, values = create_domain(2)
    solve_laplace_equation_cpu(values, numba_jacobi_solver, 1)

# Execute and display
dummy_call_to_compile_jit()
fig, ax = plt.subplots()
domain_x, domain_y, values = create_domain(NUM_NODES)

start_time = time.time()
solve_laplace_equation_cpu(values, numba_jacobi_solver, JACOBI_ITERATIONS)
#solve_laplace_equation_gpu(values, JACOBI_ITERATIONS)
end_time = time.time()
log = 'algorithm - %f sec' % (end_time - start_time)
print(log)

im = ax.pcolormesh(domain_x, domain_y, values, cmap="rainbow", antialiased=True, shading="gouraud")
fig.colorbar(im)
#fig.savefig("test.png")

