"""
@author: Vincent Bonnet
@description : Bending constraint helper functions
"""

import math
import numpy as np
import numba

from lib.objects.jit.data import Bending
import core.code_gen as generate
import lib.objects.jit.algorithms.data_accessor as db
import core.jit.math_2d as math2D
from lib.objects.jit.algorithms.differentiation_lib import force_jacobians_from_energy

@generate.vectorize
def compute_rest(bending : Bending, details):
    x0 = db.x(details.node, bending.node_IDs[0])
    x1 = db.x(details.node, bending.node_IDs[1])
    x2 = db.x(details.node, bending.node_IDs[2])
    bending.rest_angle = np.float64(math2D.angle(x0, x1, x2))

@generate.vectorize
def compute_forces(bending : Bending, details):
    x0 = db.x(details.node, bending.node_IDs[0])
    x1 = db.x(details.node, bending.node_IDs[1])
    x2 = db.x(details.node, bending.node_IDs[2])
    forces = elastic_bending_forces(x0, x1, x2, bending.rest_angle, bending.stiffness)
    bending.f[0] = forces[0]
    bending.f[1] = forces[1]
    bending.f[2] = forces[2]

@generate.vectorize
def compute_force_jacobians(bending : Bending, details):
    x0 = db.x(details.node, bending.node_IDs[0])
    x1 = db.x(details.node, bending.node_IDs[1])
    x2 = db.x(details.node, bending.node_IDs[2])
    dfdx = elastic_bending_numerical_jacobians(x0, x1, x2, bending.rest_angle, bending.stiffness)
    bending.dfdx[0][0] = dfdx[0]
    bending.dfdx[1][1] = dfdx[1]
    bending.dfdx[2][2] = dfdx[2]
    bending.dfdx[0][1] = bending.dfdx[1][0] = dfdx[3]
    bending.dfdx[0][2] = bending.dfdx[2][0] = dfdx[4]
    bending.dfdx[1][2] = bending.dfdx[2][1] = dfdx[5]

@numba.njit
def elastic_bending_energy(X, rest_angle, stiffness):
    angle = math2D.angle(X[0], X[1], X[2])
    arc_length = (math2D.norm(X[1] - X[0]) + math2D.norm(X[2] - X[1])) * 0.5
    return 0.5 * stiffness * ((angle - rest_angle)**2) * arc_length

@numba.njit
def elastic_bending_forces(x0, x1, x2, rest_angle, stiffness):
    forces = np.zeros((3, 2))

    u = x0 - x1
    v = x1 - x2
    det = u[0]*v[1] - v[0]*u[1]
    dot = u[0]*v[0] + u[1]*v[1]

    norm_u = math.sqrt(u[0]**2 + u[1]**2)
    norm_v = math.sqrt(v[0]**2 + v[1]**2)

    diff_angle = rest_angle - math.atan2(det, dot)

    forces[0][0] = v[0]*det - v[1]*dot
    forces[0][1] = v[0]*dot + v[1]*det
    forces[0] *= 0.5*(norm_u + norm_v)/(dot**2 + det**2)
    forces[0] += 0.25*u*diff_angle/norm_u
    forces[0] *= stiffness*diff_angle*-1.0

    forces[2][0] = -(u[0]*det + u[1]*dot)
    forces[2][1] = u[0]*dot - u[1]*det
    forces[2] *= 0.5*(norm_u + norm_v)/(dot**2 + det**2)
    forces[2] += -0.25*v*diff_angle/norm_v
    forces[2] *= stiffness*diff_angle*-1.0

    forces[1] -= (forces[0] + forces[2])

    return forces

@numba.njit
def elastic_bending_numerical_jacobians(x0, x1, x2, rest_angle, stiffness):
    '''
    Returns the six jacobians matrices in the following order
    df0dx0, df1dx1, df2dx2, df0dx1, df0dx2, df1dx2
    dfdx01 is the derivative of f0 relative to x1
    '''
    return force_jacobians_from_energy(x0, x1, x2, rest_angle, stiffness, elastic_bending_energy)
