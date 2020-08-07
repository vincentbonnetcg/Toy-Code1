"""
@author: Vincent Bonnet
@description : Area constraint helper functions
"""

import math
import numpy as np
import numba
import lib.common.jit.math_2d as math2D
from lib.objects.jit.utils.differentiation_lib import force_jacobians_from_energy

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

