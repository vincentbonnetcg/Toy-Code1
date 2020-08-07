"""
@author: Vincent Bonnet
@description : functions to perform numerical differentiations
useful to check the analytic differentiations
"""
import numba
import numpy as np
import lib.common.jit.math_2d as math2D

@numba.njit
def force_jacobians_from_energy(x0, x1, x2, rest_value, stiffness, energy_func):
    '''
    Returns the six jacobians matrices in the following order
    df0dx0, df1dx1, df2dx2, df0dx1, df0dx2, df1dx2
    for example : dfdx01 is the derivative of f0 relative to x1
    TODO : performance could be improved
    '''
    jacobians = np.zeros(shape=(6, 2, 2))
    STENCIL_SIZE = 1e-6

    E = np.empty(shape=(3,3))
    # df0dx0, df1dx1, df2dx2, df0dx1, df0dx2, df1dx2
    arg_indices = [[0,0],[1,1],[2,2],[0,1],[0,2],[1,2]] # argument indices
    el_indices = [[0,0],[0,1],[1,1]] # element indices
    for arg_index in range(6):

        arg_ids = arg_indices[arg_index]

        if arg_ids[0]==arg_ids[1]:
            # Special case which reduces from 12 to 9 energy computations
            # collect energy from stencils
            #  indices(idx)     stencil offsets
            #   0 1 2          -t,t   0,t   t,t
            #   3 4 5     =>   -t,0   0,0   t,0
            #   6 7 8          -t,-t  0,-t  t,-t
            for idx in range(9):
                X = [math2D.copy(x0), math2D.copy(x1), math2D.copy(x2)] # TODO - slow
                i = idx%3
                j = int((idx-i)/3)
                X[arg_ids[0]][0] += (i-1)*STENCIL_SIZE
                X[arg_ids[1]][1] += (1-j)*STENCIL_SIZE
                E[i,j] = energy_func(X, rest_value, stiffness)

            # compute second derivates of the energy
            ded00 = (E[0,1]+E[2,1]-(E[1,1]*2.0)) / STENCIL_SIZE**2
            ded11 = (E[1,0]+E[1,2]-(E[1,1]*2.0)) / STENCIL_SIZE**2
            ded01 = (E[2,0]+E[0,2]-E[0,0]-E[2,2]) / (4.0 * STENCIL_SIZE**2)
            # assemble the jacobian forces
            jacobians[arg_index,0,0] = -ded00
            jacobians[arg_index,1,1] = -ded11
            jacobians[arg_index,0,1] = -ded01
            jacobians[arg_index,1,0] = -ded01 # from Schwarz's theorem
        else:
            for el_index in range(3):
                ii = el_indices[el_index][0]
                jj = el_indices[el_index][1]
                # collect energy from stencils
                # indices(idx)    stencil offsets
                #  0  1            -t,t    t,t
                #  2  3            -t,-t   t,-t
                for idx in range(4):
                    X = [math2D.copy(x0), math2D.copy(x1), math2D.copy(x2)] # TODO - slow
                    i = (idx%2)
                    j = int((idx-i)/2)
                    X[arg_ids[0]][ii] += (i*2-1)*STENCIL_SIZE
                    X[arg_ids[1]][jj] += (1-j*2)*STENCIL_SIZE
                    E[i,j] = energy_func(X, rest_value, stiffness)
                # compute second derivate of the energy
                dedij = (E[1,0]+E[0,1]-E[0,0]-E[1,1]) / (4.0 * STENCIL_SIZE**2)
                # assemble the jacobian forces
                jacobians[arg_index,ii,jj] = -dedij
            # from Schwarz's theorem
            jacobians[arg_index,1,0] = jacobians[arg_index,0,1]

    return jacobians
