"""
@author: Vincent Bonnet
@description : Area constraint helper functions
"""

import math
import numpy as np
import numba

from lib.objects.jit.data import Area
import core.code_gen as generate
import core.jit.data_accessor as db
import core.jit.math_2d as math2D
from lib.objects.jit.algorithms.differentiation_lib import force_jacobians_from_energy

@generate.vectorize
def compute_rest(area : Area, details):
    x0 = db.x(details.node, area.node_IDs[0])
    x1 = db.x(details.node, area.node_IDs[1])
    x2 = db.x(details.node, area.node_IDs[2])
    area.rest_area = np.float64(math2D.area(x0, x1, x2))

@generate.vectorize
def compute_forces(area : Area, details):
    x0 = db.x(details.node, area.node_IDs[0])
    x1 = db.x(details.node, area.node_IDs[1])
    x2 = db.x(details.node, area.node_IDs[2])
    forces = elastic_area_forces(x0, x1, x2, area.rest_area, area.stiffness)
    area.f[0] = forces[0]
    area.f[1] = forces[1]
    area.f[2] = forces[2]

@generate.vectorize
def compute_force_jacobians(area : Area, details):
    x0 = db.x(details.node, area.node_IDs[0])
    x1 = db.x(details.node, area.node_IDs[1])
    x2 = db.x(details.node, area.node_IDs[2])
    jacobians = elastic_area_numerical_jacobians(x0, x1, x2, area.rest_area, area.stiffness)
    area.dfdx[0][0] = jacobians[0]
    area.dfdx[1][1] = jacobians[1]
    area.dfdx[2][2] = jacobians[2]
    area.dfdx[0][1] = area.dfdx[1][0] = jacobians[3]
    area.dfdx[0][2] = area.dfdx[2][0] = jacobians[4]
    area.dfdx[1][2] = area.dfdx[2][1] = jacobians[5]

@numba.njit
def elastic_area_energy(X, rest_area, stiffness):
    # X => [x0, x1, x2]
    area = math2D.area(X[0], X[1], X[2])
    return 0.5 * stiffness * ((area - rest_area)**2)

@numba.njit
def elastic_area_forces(x0, x1, x2, rest_area, stiffness):
    forces = np.zeros((3, 2))

    u = x0 - x1
    v = x1 - x2
    w = x0 - x2
    det = u[0]*w[1] - w[0]*u[1]
    tmp = 0.5*stiffness*(rest_area - math.fabs(det) * 0.5)*np.sign(det)

    forces[0][0] = v[1]
    forces[0][1] = v[0] * -1.0
    forces[0] *= tmp

    forces[2][0] = u[1]
    forces[2][1] = u[0] * -1.0
    forces[2] *= tmp

    forces[1] -= (forces[0] + forces[2])

    return forces

@numba.njit
def elastic_area_numerical_jacobians(x0, x1, x2, rest_area, stiffness):
    '''
    Returns the six jacobians matrices in the following order
    df0dx0, df1dx1, df2dx2, df0dx1, df0dx2, df1dx2
    dfdx01 is the derivative of f0 relative to x1
    '''
    return force_jacobians_from_energy(x0, x1, x2, rest_area, stiffness, elastic_area_energy)

