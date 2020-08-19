"""
@author: Vincent Bonnet
@description : Spring constraint helper functions
"""

import numpy as np
import numba

from lib.objects.jit.data import Spring
import lib.common.code_gen as generate
import lib.common.jit.data_accessor as db
import lib.common.jit.math_2d as math2D

@generate.vectorize
def compute_rest(spring : Spring, details):
    x0 = db.x(details.node, spring.node_IDs[0])
    x1 = db.x(details.node, spring.node_IDs[1])
    spring.rest_length = np.float64(math2D.distance(x0, x1))

@generate.vectorize
def compute_forces(spring : Spring, details):
    x0, v0 = db.xv(details.node, spring.node_IDs[0])
    x1, v1 = db.xv(details.node, spring.node_IDs[1])
    force = spring_stretch_force(x0, x1, spring.rest_length, spring.stiffness)
    force += spring_damping_force(x0, x1, v0, v1, spring.damping)
    spring.f[0] = force
    spring.f[1] = force * -1.0

@generate.vectorize
def compute_force_jacobians(spring : Spring, details):
    x0, v0 = db.xv(details.node, spring.node_IDs[0])
    x1, v1 = db.xv(details.node, spring.node_IDs[1])
    dfdx = spring_stretch_jacobian(x0, x1, spring.rest_length, spring.stiffness)
    dfdv = spring_damping_jacobian(x0, x1, v0, v1, spring.damping)
    spring.dfdx[0][0] = spring.dfdx[1][1] = dfdx
    spring.dfdx[0][1] = spring.dfdx[1][0] = dfdx * -1
    spring.dfdv[0][0] = spring.dfdv[1][1] = dfdv
    spring.dfdv[0][1] = spring.dfdv[1][0] = dfdv * -1

'''
AnchorSpring/Spring helper functions
'''
@numba.njit
def spring_stretch_jacobian(x0, x1, rest, stiffness):
    direction = x0 - x1
    stretch = math2D.norm(direction)
    I = np.identity(2)
    if not math2D.is_close(stretch, 0.0):
        direction /= stretch
        A = np.outer(direction, direction)
        return -1.0 * stiffness * ((1 - (rest / stretch)) * (I - A) + A)

    return -1.0 * stiffness * I

@numba.njit
def spring_damping_jacobian(x0, x1, v0, v1, damping):
    jacobian = np.zeros(shape=(2, 2))
    direction = x1 - x0
    stretch = math2D.norm(direction)
    if not math2D.is_close(stretch, 0.0):
        direction /= stretch
        A = np.outer(direction, direction)
        jacobian = -1.0 * damping * A

    return jacobian

@numba.njit
def spring_stretch_force(x0, x1, rest, stiffness):
    direction = x1 - x0
    stretch = math2D.norm(direction)
    if not math2D.is_close(stretch, 0.0):
        direction /= stretch
    return direction * ((stretch - rest) * stiffness)

@numba.njit
def spring_damping_force(x0, x1, v0, v1, damping):
    direction = x1 - x0
    stretch = math2D.norm(direction)
    if not math2D.is_close(stretch, 0.0):
        direction /= stretch
    relativeVelocity = v1 - v0
    return direction * (np.dot(relativeVelocity, direction) * damping)

@numba.njit
def elastic_spring_energy(x0, x1, rest, stiffness):
    stretch = math2D.distance(x0, x1)
    return 0.5 * stiffness * ((stretch - rest)**2)
