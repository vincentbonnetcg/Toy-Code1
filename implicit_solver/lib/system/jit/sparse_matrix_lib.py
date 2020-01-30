"""
@author: Vincent Bonnet
@description : Helper functions for sparse matrix assembly
"""

import numba
import numpy as np

@numba.njit
def create_empty_sparse_matrix(num_rows, block_size):
    A = []
    for i in range(num_rows):
        A.append({i:np.zeros((block_size,block_size))})
    return A

@numba.njit
def add(A, i, j, data):
    if j in A[i]:
        A[i][j] += data
    else:
        A[i][j] = data
