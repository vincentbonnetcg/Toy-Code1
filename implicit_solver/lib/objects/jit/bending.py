"""
@author: Vincent Bonnet
@description : Bending Constraint for the implicit solver
"""

import numpy as np
import numba # required by lib.common.code_gen

import lib.common.jit.math_2d as math2D
import lib.common.jit.data_accessor as db
import lib.common.code_gen as generate
import lib.objects.jit.utils.bending_lib as bending_lib
from lib.objects.jit import Constraint

class Bending(Constraint):
    '''
    Describes a 2D bending constraint of a thin inextensible wire
    between three nodes.
    This bending is NOT the proper bending formulation and uses angle instead of curvature
    Some instabilities when using the curvature => Need to investigate
    '''
    def __init__(self):
        '''
        Constraint three nodes to maintain angle between
        node_ids[0] - node_ids[1] - node_ids[2]
        '''
        Constraint.__init__(self, num_nodes = 3)
        self.rest_angle = np.float64(0.0)

    # initialization functions
    @classmethod
    def pre_compute(cls):
        return None

    @classmethod
    def compute_rest(cls):
        return compute_bending_rest

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
        return compute_bending_forces

    @classmethod
    def compute_force_jacobians(cls):
        return compute_bending_force_jacobians

@generate.as_vectorized(block_handles=True)
def compute_bending_rest(bending : Bending, detail_nodes):
    x0 = db.x(detail_nodes, bending.node_IDs[0])
    x1 = db.x(detail_nodes, bending.node_IDs[1])
    x2 = db.x(detail_nodes, bending.node_IDs[2])
    bending.rest_angle = np.float64(math2D.angle(x0, x1, x2))

@generate.as_vectorized(block_handles=True)
def compute_bending_forces(bending : Bending, detail_nodes):
    x0 = db.x(detail_nodes, bending.node_IDs[0])
    x1 = db.x(detail_nodes, bending.node_IDs[1])
    x2 = db.x(detail_nodes, bending.node_IDs[2])
    forces = bending_lib.elastic_bending_forces(x0, x1, x2, bending.rest_angle, bending.stiffness)
    bending.f[0] = forces[0]
    bending.f[1] = forces[1]
    bending.f[2] = forces[2]

@generate.as_vectorized(block_handles=True)
def compute_bending_force_jacobians(bending : Bending, detail_nodes):
    x0 = db.x(detail_nodes, bending.node_IDs[0])
    x1 = db.x(detail_nodes, bending.node_IDs[1])
    x2 = db.x(detail_nodes, bending.node_IDs[2])
    dfdx = bending_lib.elastic_bending_numerical_jacobians(x0, x1, x2, bending.rest_angle, bending.stiffness)
    bending.dfdx[0][0] = dfdx[0]
    bending.dfdx[1][1] = dfdx[1]
    bending.dfdx[2][2] = dfdx[2]
    bending.dfdx[0][1] = bending.dfdx[1][0] = dfdx[3]
    bending.dfdx[0][2] = bending.dfdx[2][0] = dfdx[4]
    bending.dfdx[1][2] = bending.dfdx[2][1] = dfdx[5]
