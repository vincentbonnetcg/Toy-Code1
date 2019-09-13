"""
@author: Vincent Bonnet
@description : Lattice Boltzmann Method (D2Q9 Model)

# Lattice Velocities for f_in and f_out#
f_in : Incoming Population
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

f_out : Outgoing Population
----------------
| V8   v4  v7  |
|   \  |  /    |
|    \ | /     |
|     \|/      |
| v1---v0---v3 |
|    / | \     |
|   /  |  \    |
|  /   |   \   |
|v5    v2   v6 |
----------------
v0 = (0,0) ,
v1 = (-1,0), v2 = (0,-1), v3 = (1,0), v4 = (0,1)
v5 = (-1,-1), v6 = (1,-1), v7 = (1,1), v8 = (-1,1)

------------------

# Variables #
f_in : incoming popution on a cell (9 values per cell) (see Lattice Velocities)
f_out : outcoming population on a cell (9 values per cell) (see Lattice Velocities)
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

# Collision Step #
BJK Collision Model
i is index from 0 to 8
fi_out - fi_in = -omega * (f_in - E(i, d, u))
so the population after the collision is :
fi_out = fi_in - omega * (f_in - E(i, d, u))

# Streaming Step #
Move data from one cell to another
"""

import numpy as np
import time
import numba
import matplotlib.pyplot as plt
from matplotlib import cm

NUM_CELL_X = 64
NUM_CELL_Y = 64
OMEGA = 1.0 # TODO - need to express from Reynold Number
NUM_ITERATIONS = 10

LATTICE_Vf = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=np.float64)
LATTICE_Vi = np.array([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]], dtype=np.int64)
T = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
OPPOSITE = [0,3,4,1,2,7,8,5,6] # see f_in or f_out


def macroscopic_variables(f_in):
    '''
    computes the density and momentum
    '''
    d = np.zeros((NUM_CELL_X, NUM_CELL_Y), dtype=np.float64) # density
    u = np.zeros((2, NUM_CELL_X, NUM_CELL_Y), dtype=np.float64) # velocity

    # compute density from incoming population (f_in)
    # for each cell sum 'incoming population'
    np.sum(f_in, axis=0, out = d)

    # compute velocities from incoming population (f_in)
    # for each cell sum LATTICE_VELOCITIES weighted with 'incoming population'
    for v_id in range(9):
        u[0,:,:] += LATTICE_Vf[v_id, 0] * f_in[v_id,:,:] # compute v.x
        u[1,:,:] += LATTICE_Vf[v_id, 1] * f_in[v_id,:,:] # compute v.y
    u /= d

    return d, u

def equilibrium_distribution(d, u):
    '''
    compute equilibrium distribution from density and momentum
    '''
    e = np.zeros((9, NUM_CELL_X, NUM_CELL_Y), dtype=np.float64) # equilibrium distribution
    # E(i,d,u) = d * T[i] * (1 + (3 * (vi.u)) + ( 0.5 * (3 * (vi.u))^2 ) + 3/2*|u|^2 )
    dot_u = 3.0/2.0 * u[0]**2+u[1]**2 # 3/2*|u|^2
    for v_id in range(9):
        vu = 3.0 * (LATTICE_Vf[v_id, 0] * u[0,:,:] + LATTICE_Vf[v_id, 1] * u[1,:,:] )
        e[v_id,:,:] = d * T[v_id] * (1.0 + vu + 0.5 * vu ** 2 - dot_u)

    return e

@numba.njit
def stream_step(f_in, f_out):
    '''
    stream operation
    '''
    for i in range(NUM_CELL_X):
        for j in range(NUM_CELL_Y):
            for v_id in range(9):
                next_i = i + LATTICE_Vi[v_id,0]
                next_j = j + LATTICE_Vi[v_id,1]
                # boundary condition in X axis (periodic boundary)
                if next_i < 0:
                    next_i = NUM_CELL_X-1
                elif next_i > NUM_CELL_X-1:
                    next_i = 0
                # boundary condition in Y axis (periodic boundary)
                if next_j < 0:
                    next_j = NUM_CELL_Y-1
                elif next_j > NUM_CELL_Y-1:
                    next_j = 0
                # move out to incoming population
                f_in[v_id, next_i, next_j] = f_out[v_id, i, j]

def solid_boundary_func(x, y):
    '''
    solid boundary is a binary mask (0 no solid, 1 solid)
    '''
    cx, cy = NUM_CELL_X/2, NUM_CELL_Y/2
    r = NUM_CELL_Y / 6
    return (x-cx)**2+(y-cy)**2<r**2

def lbm(f_in, solid_boundary):

    f_out = np.zeros((9, NUM_CELL_X, NUM_CELL_Y), dtype=np.float64) # outgoing population

    # add boundary flow condition
    # TODO

    # compute density and momentum from incoming population (f_in)
    d, u = macroscopic_variables(f_in)

    # compute equilibrium distribution from density and velocities
    e = equilibrium_distribution(d, u)

    # compute population after collision
    # fi_out = fi_in - omega * (f_in - E(i, d, u))
    f_out = f_in - OMEGA * (f_in - e)

    # enforce solid boundary with bounce-back technique
    for v_id in range(9):
        f_out[v_id, solid_boundary] = f_in[OPPOSITE[v_id], solid_boundary]

    # compute population after the stream step
    stream_step(f_in, f_out)

    return u

if __name__ == '__main__':
    solid_boundary = np.fromfunction(solid_boundary_func, (NUM_CELL_X, NUM_CELL_Y)) # solid boundary
    f_in = np.ones((9, NUM_CELL_X, NUM_CELL_Y), dtype=np.float64) * 0.001 # incoming population
    f_in[1].fill(0.01)

    for _ in range(NUM_ITERATIONS):
        u = lbm(f_in, solid_boundary)

        # show result with the velocity norm
        u_norm = np.sqrt(u[0]**2+u[1]**2)
        plt.imshow(u_norm.transpose(), cmap=cm.Blues)
        plt.show()
        #plt.imshow(solid_boundary.astype(np.float64), cmap=cm.Blues)
        #plt.show()
