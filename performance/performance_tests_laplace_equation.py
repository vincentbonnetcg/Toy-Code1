"""
@author: Vincent Bonnet
@description : Evaluate CPU and stuff
"""

import time
import numpy as np
from numba import njit, cuda, vectorize
import matplotlib.pyplot as plt
import matplotlib as mpl

'''
 Global Parameters
'''
NUM_NODES = 50
JACOBI_ITERATIONS = 500

# TODO - add time + graph
# TODO - add numba cuda
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
        for j in range(1, num_dof_y - 1):
            for i in range(1, num_dof_x - 1):
                next_x[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1]) * 0.25

        np.copyto(x, next_x)

@njit
def numba_jacobi_solver(x, next_x,  num_iterations = 500):
    num_dof_x = x.shape[0]
    num_dof_y = x.shape[1]

    for iteration in range(num_iterations):
        for j in range(1, num_dof_y - 1):
            for i in range(1, num_dof_x - 1):
                next_x[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1]) * 0.25

        # numba doesn't support copyto
        #np.copyto(x, next_x)
        for j in range(1, num_dof_y - 1):
            for i in range(1, num_dof_x - 1):
                x[i][j] = next_x[i][j]

def solve_laplace_equation(x, algo = jacobi_solver, num_iterations = 500):
    next_x = np.copy(x)
    for iteration in range(0, num_iterations, 1):
        algo(x, next_x, 1)

# Execute and display
fig, ax = plt.subplots()
domain_x, domain_y, values = create_domain(NUM_NODES)

start_time = time.time()
solve_laplace_equation(values, numba_jacobi_solver, JACOBI_ITERATIONS)
end_time = time.time()
log = ' %f sec' % (end_time - start_time)
print(log)

im = ax.pcolormesh(domain_x, domain_y, values, cmap="rainbow", antialiased=True, shading="gouraud")
fig.colorbar(im)
#fig.savefig("test.png")

