"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np
import numba # required by lib.common.code_gen

import lib.common.jit.math_2d as math2D
import lib.common.jit.node_accessor as na
import lib.common.code_gen as generate
import lib.objects.jit.utils.spring_lib as spring_lib
from lib.objects.jit import Constraint

class AnchorSpring(Constraint):
    '''
    Describes a 2D spring constraint between a node and point
    '''
    def __init__(self):
        Constraint.__init__(self, num_nodes = 1)
        self.rest_length = np.float64(0.0)
        self.kinematic_component_IDs = na.empty_node_ids(2) # Point ids
        self.kinematic_component_param = np.float64(0.0)
        self.kinematic_component_pos = np.zeros(2, dtype = np.float64)

    # initialization functions
    @classmethod
    def pre_compute(cls):
        return pre_compute_anchor_spring

    @classmethod
    def compute_rest(cls):
        return compute_anchor_spring_rest

    # constraint functions (cost, gradients, hessians)
    @classmethod
    def compute_cost(cls):
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
        return compute_anchor_spring_forces

    @classmethod
    def compute_force_jacobians(cls):
        return compute_anchor_spring_force_jacobians

@generate.as_vectorized(block_handles=True)
def pre_compute_anchor_spring(anchor_spring : AnchorSpring, detail_nodes, details_points):
    t = anchor_spring.kinematic_component_param
    x0 = na.node_x(details_points, anchor_spring.kinematic_component_IDs[0])
    x1 = na.node_x(details_points, anchor_spring.kinematic_component_IDs[1])
    anchor_spring.kinematic_component_pos = x0 * (1.0 - t) + x1 * t

@generate.as_vectorized(block_handles=True)
def compute_anchor_spring_rest(anchor_spring : AnchorSpring, detail_nodes):
    x = na.node_x(detail_nodes, anchor_spring.node_IDs[0])
    anchor_spring.rest_length = np.float64(math2D.distance(anchor_spring.kinematic_component_pos, x))

@generate.as_vectorized(block_handles=True)
def compute_anchor_spring_forces(anchor_spring : AnchorSpring, detail_nodes):
    x, v = na.node_xv(detail_nodes, anchor_spring.node_IDs[0])
    kinematic_vel = np.zeros(2)
    target_pos = anchor_spring.kinematic_component_pos
    force = spring_lib.spring_stretch_force(x, target_pos, anchor_spring.rest_length, anchor_spring.stiffness)
    force += spring_lib.spring_damping_force(x, target_pos, v, kinematic_vel, anchor_spring.damping)
    anchor_spring.f = force

@generate.as_vectorized(block_handles=True)
def compute_anchor_spring_force_jacobians(anchor_spring : AnchorSpring, detail_nodes):
    x, v = na.node_xv(detail_nodes, anchor_spring.node_IDs[0])
    kinematic_vel = np.zeros(2)
    target_pos = anchor_spring.kinematic_component_pos
    dfdx = spring_lib.spring_stretch_jacobian(x, target_pos, anchor_spring.rest_length, anchor_spring.stiffness)
    dfdv = spring_lib.spring_damping_jacobian(x, target_pos, v, kinematic_vel, anchor_spring.damping)
    anchor_spring.dfdx[0][0] = dfdx
    anchor_spring.dfdv[0][0] = dfdv
