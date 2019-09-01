"""
@author: Vincent Bonnet
@description : Lattice Boltzmann Method (D2Q9 Model)

# Lattice Velocities #
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

# Variables #
f_in : incoming popution on a cell (9 values per cell)
f_out : outcoming population on a cell (9 values per cell)
d : density per cell (single value per cell)
u : average velocity per cell (2 dimension array per cell)

# Parameters #
w (omega) : relaxation parameter defining how fast a fluid converges to its equlibrium
        high omega => low viscosity, low omega => high viscosity

# Pressure is proportional to density #
p = C^2 * d
where C^2 = 1/3 * (dx*dx/dt*dt)

# Equilibrium Distribition Formula #
E(i,d,u) = d * T[i] * (1 + ( (vi.u) / C^2) + (1/(2*C^4) * (vi.u)^2) - 1/(C^2)*|u|^2 )
where T = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36]
      and C^2 = 1/3 * (dx*dx/dt*dt)

Using in dx=1 and dt=1 (Lattice Units) , we can simplify E(i,d, u)
E(i,d,u) = d * T[i] * (1 + (3 * (vi.u)) + ( 0.5 * (3 * (vi.u))^2 ) + 3/2*|u|^2 )

# Collision Formula #
BJK Collision Model
i is index from 0 to 8
fi_out - fi_in = -omega * (f_in - E(i, d, u))
so the population after the collision is :
fi_out = fi_in - omega * (f_in - E(i, d, u))

"""

import numpy as np
import time
import numba

NUM_CELL_X = 4
NUM_CELL_Y = 4
EPSILON = 0.001 # prevent division by zero
OMEGA = 1.0 # TODO - need to express from Reynold Number

LATTICE_V = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=np.float64)
T = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)

def lbm(f_in):
    d = np.zeros((NUM_CELL_X, NUM_CELL_Y), dtype=np.float64) # density
    u = np.zeros((NUM_CELL_X, NUM_CELL_Y, 2), dtype=np.float64) # velocity
    e = np.zeros((NUM_CELL_X, NUM_CELL_Y, 9), dtype=np.float64) # equilibrium distribution
    f_out = np.zeros((NUM_CELL_X, NUM_CELL_Y, 9), dtype=np.float64) # outgoing population

    # compute density
    # for each cell sum 'incoming population'
    np.sum(f_in, axis=2, out = d)

    # compute velocities
    # for each cell sum LATTICE_VELOCITIES weighted with 'incoming population'
    # Explicit Loop
    #for i in range(NUM_CELL_X):
    #    for j in range(NUM_CELL_Y):
    #        for k in range(9):
    #            u[i, j] += LATTICE_VELOCITIES[k] * f_in[i, j, k]
    #        u[i, j] /= d[i, j]
    # Numpy Vectorized
    for v_id in range(9):
        u[:,:,0] += LATTICE_V[v_id, 0] * f_in[:,:,v_id] # compute v.x
        u[:,:,1] += LATTICE_V[v_id, 1] * f_in[:,:,v_id] # compute v.y

    u[:,:,0] /= d
    u[:,:,1] /= d

    # compute equilibrium distribution
    # E(i,d,u) = d * T[i] * (1 + (3 * (vi.u)) + ( 0.5 * (3 * (vi.u))^2 ) + 3/2*|u|^2 )
    dot_u = 3.0/2.0 * u[:,:,0]**2+u[:,:,1]**2 # 3/2*|u|^2
    for v_id in range(9):
        vu = 3.0 * (LATTICE_V[v_id, 0] * u[:,:,0] + LATTICE_V[v_id, 1] * u[:,:,1] )
        e[:,:,v_id] = d * T[v_id] * (1.0 + vu + 0.5 * vu ** 2 - dot_u)

    # compute population after collsiion
    # fi_out = fi_in - omega * (f_in - E(i, d, u))
    f_out = f_in - OMEGA * (f_in - e)


f_in = np.ones((NUM_CELL_X, NUM_CELL_Y, 9), dtype=np.float64) * EPSILON # incoming population
#f_in = np.random.rand(NUM_CELL_X, NUM_CELL_Y, 9)
lbm(f_in)

