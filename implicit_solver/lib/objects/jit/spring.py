"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np
import numba # required by lib.common.code_gen

import lib.common.jit.math_2d as math2D
import lib.common.jit.data_accessor as db
import lib.common.code_gen as generate
import lib.objects.jit.utils.spring_lib as spring_lib
from lib.objects.jit import Constraint

class Spring(Constraint):
    '''
    Describes a 2D spring constraint between two nodes
    '''
    def __init__(self):
        Constraint.__init__(self, num_nodes = 2)
        self.rest_length = np.float64(0.0)

    # initialization functions
    @classmethod
    def pre_compute(cls):
        return None

    @classmethod
    def compute_rest(cls):
        return compute_spring_rest

    # constraint functions (function, gradients, hessians)
    @classmethod
    def compute_function(cls):
        return None # TODO

    @classmethod
    def compute_gradients(cls):
        return None # TODO

    @classmethod
    def compute_hessians(cls):
        return None # TODO

    # force functions (forces and their jacobians)
    @classmethod
    def compute_forces(cls):
        return compute_spring_forces

    @classmethod
    def compute_force_jacobians(cls):
        return compute_spring_force_jacobians

@generate.as_vectorized(block_handles=True)
def compute_spring_rest(spring : Spring, detail_nodes):
    x0 = db.x(detail_nodes, spring.node_IDs[0])
    x1 = db.x(detail_nodes, spring.node_IDs[1])
    spring.rest_length = np.float64(math2D.distance(x0, x1))

@generate.as_vectorized(block_handles=True)
def compute_spring_forces(spring : Spring, detail_nodes):
    x0, v0 = db.xv(detail_nodes, spring.node_IDs[0])
    x1, v1 = db.xv(detail_nodes, spring.node_IDs[1])
    force = spring_lib.spring_stretch_force(x0, x1, spring.rest_length, spring.stiffness)
    force += spring_lib.spring_damping_force(x0, x1, v0, v1, spring.damping)
    spring.f[0] = force
    spring.f[1] = force * -1.0

@generate.as_vectorized(block_handles=True)
def compute_spring_force_jacobians(spring : Spring, detail_nodes):
    x0, v0 = db.xv(detail_nodes, spring.node_IDs[0])
    x1, v1 = db.xv(detail_nodes, spring.node_IDs[1])
    dfdx = spring_lib.spring_stretch_jacobian(x0, x1, spring.rest_length, spring.stiffness)
    dfdv = spring_lib.spring_damping_jacobian(x0, x1, v0, v1, spring.damping)
    spring.dfdx[0][0] = spring.dfdx[1][1] = dfdx
    spring.dfdx[0][1] = spring.dfdx[1][0] = dfdx * -1
    spring.dfdv[0][0] = spring.dfdv[1][1] = dfdv
    spring.dfdv[0][1] = spring.dfdv[1][0] = dfdv * -1

