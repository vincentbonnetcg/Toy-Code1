"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np

import lib.common.jit.math_2d as math2D
import lib.common.jit.node_accessor as na
import lib.common.code_gen as generate
import lib.common.convex_hull as ch
import lib.objects.components_jit.utils.spring_lib as spring_lib
import lib.objects.components_jit as cpn

class AnchorSpring(cpn.ConstraintBase):
    '''
    Describes a 2D spring constraint between a node and point
    '''
    def __init__(self):
        cpn.ConstraintBase.__init__(self, num_nodes = 1)
        self.rest_length = np.float64(0.0)
        self.kinematic_index = np.uint32(0)
        self.kinematic_component_index =  np.uint32(0)
        self.kinematic_component_param = np.float64(0.0)
        self.kinematic_component_pos = np.zeros(2, dtype = np.float64)

    @classmethod
    def pre_compute(cls):
        return pre_compute_anchor_spring

    @classmethod
    def compute_rest(cls):
        return compute_anchor_spring_rest

    @classmethod
    def compute_gradients(cls):
        return compute_anchor_spring_forces

    @classmethod
    def compute_hessians(cls):
        return compute_anchor_spring_jacobians

class Spring(cpn.ConstraintBase):
    '''
    Describes a 2D spring constraint between two nodes
    '''
    def __init__(self):
        cpn.ConstraintBase.__init__(self, num_nodes = 2)
        self.rest_length = np.float64(0.0)

    @classmethod
    def pre_compute(cls):
        return None

    @classmethod
    def compute_rest(cls):
        return compute_spring_rest

    @classmethod
    def compute_gradients(cls):
        return compute_spring_forces

    @classmethod
    def compute_hessians(cls):
        return compute_spring_jacobians

'''
AnchorSpring compute functions
'''
@generate.as_vectorized(njit=False, block_handles=True)
def pre_compute_anchor_spring(anchor_spring : AnchorSpring, scene, detail_nodes):
    kinematic = scene.kinematics[anchor_spring.kinematic_index]
    point_params = ch.ConvexHull.ParametricPoint(anchor_spring.kinematic_component_index, anchor_spring.kinematic_component_param)
    anchor_spring.kinematic_component_pos = kinematic.get_position_from_parametric_point(point_params)

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
def compute_anchor_spring_jacobians(anchor_spring : AnchorSpring, detail_nodes):
    x, v = na.node_xv(detail_nodes, anchor_spring.node_IDs[0])
    kinematic_vel = np.zeros(2)
    target_pos = anchor_spring.kinematic_component_pos
    dfdx = spring_lib.spring_stretch_jacobian(x, target_pos, anchor_spring.rest_length, anchor_spring.stiffness)
    dfdv = spring_lib.spring_damping_jacobian(x, target_pos, v, kinematic_vel, anchor_spring.damping)
    anchor_spring.dfdx[0][0] = dfdx
    anchor_spring.dfdv[0][0] = dfdv

'''
Spring compute functions
'''
@generate.as_vectorized(block_handles=True)
def compute_spring_rest(spring : Spring, detail_nodes):
    x0 = na.node_x(detail_nodes, spring.node_IDs[0])
    x1 = na.node_x(detail_nodes, spring.node_IDs[1])
    spring.rest_length = np.float64(math2D.distance(x0, x1))

@generate.as_vectorized(block_handles=True)
def compute_spring_forces(spring : Spring, detail_nodes):
    x0, v0 = na.node_xv(detail_nodes, spring.node_IDs[0])
    x1, v1 = na.node_xv(detail_nodes, spring.node_IDs[1])
    force = spring_lib.spring_stretch_force(x0, x1, spring.rest_length, spring.stiffness)
    force += spring_lib.spring_damping_force(x0, x1, v0, v1, spring.damping)
    spring.f[0] = force
    spring.f[1] = force * -1.0

@generate.as_vectorized(block_handles=True)
def compute_spring_jacobians(spring : Spring, detail_nodes):
    x0, v0 = na.node_xv(detail_nodes, spring.node_IDs[0])
    x1, v1 = na.node_xv(detail_nodes, spring.node_IDs[1])
    dfdx = spring_lib.spring_stretch_jacobian(x0, x1, spring.rest_length, spring.stiffness)
    dfdv = spring_lib.spring_damping_jacobian(x0, x1, v0, v1, spring.damping)
    spring.dfdx[0][0] = spring.dfdx[1][1] = dfdx
    spring.dfdx[0][1] = spring.dfdx[1][0] = dfdx * -1
    spring.dfdv[0][0] = spring.dfdv[1][1] = dfdv
    spring.dfdv[0][1] = spring.dfdv[1][0] = dfdv * -1

