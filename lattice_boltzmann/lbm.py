"""
@author: Vincent Bonnet
@description : Lattice Boltzmann Method (D2Q9 Model)

Lattice Velocities
----------------
| V6   v2  v5  |
|   \  |  /    |
|    \ | /     |
|     \|/      |
| v3---v0---v1 |
|    / | \     |
|   /  |  \    |
|  /   |   \   |
|v7    v4   v8 |
----------------
v0 = (0,0) ,
v1 = (1,0), v2 = (0,1), v3 = (-1,0), v4 = (0,-1)
v5 = (1,1), v6 = (-1,1), v7 = (-1,-1), v8 = (1,-1)
------------------
"""

import numpy as np
import time
import numba

NUM_CELL_X = 16
NUM_CELL_Y = 16

LATTICE_VELOCITIES = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=np.float64)

def lbm(f_in, f_out, d, v):
    # compute density
    # for each cell sum 'incoming population'
    np.sum(f_in, axis=2, out = d)

    # compute velocities
    # for each cell sum LATTICE_VELOCITIES weighted with 'incoming population'
    # Explicit Loop
    #for i in range(NUM_CELL_X):
    #    for j in range(NUM_CELL_Y):
    #        for k in range(9):
    #            v[i, j] += LATTICE_VELOCITIES[k] * f_in[i, j, k]
    #        v[i, j] /= d[i, j]
    # Numpy Vectorized
    for v_id in range(9):
        v[:,:,0] += LATTICE_VELOCITIES[v_id, 0] * f_in[:,:,v_id] # compute v.x
        v[:,:,1] += LATTICE_VELOCITIES[v_id, 1] * f_in[:,:,v_id] # compute v.y
    v[:,:,0] /= d
    v[:,:,1] /= d


f_in = np.zeros((NUM_CELL_X, NUM_CELL_Y, 9), dtype=np.float64) # incoming population
f_out = np.zeros((NUM_CELL_X, NUM_CELL_Y, 9), dtype=np.float64) # outgoing population
d = np.zeros((NUM_CELL_X, NUM_CELL_Y), dtype=np.float64) # density
v = np.zeros((NUM_CELL_X, NUM_CELL_Y, 2), dtype=np.float64) # velocity


lbm(f_in, f_out, d, v)



