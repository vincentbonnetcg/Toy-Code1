"""
@author: Vincent Bonnet
@description : AnchorSpring constraint helper functions
"""
import numpy as np
import numba

from lib.objects.jit.data import AnchorSpring
import lib.objects.jit.algorithms.spring_lib as spring_lib
import lib.common.code_gen as generate
import lib.common.jit.data_accessor as db
import lib.common.jit.math_2d as math2D

@generate.vectorize
def pre_compute(anchor_spring : AnchorSpring, details):
    t = anchor_spring.kinematic_component_param
    x0 = db.x(details.point, anchor_spring.kinematic_component_IDs[0])
    x1 = db.x(details.point, anchor_spring.kinematic_component_IDs[1])
    anchor_spring.kinematic_component_pos = x0 * (1.0 - t) + x1 * t

@generate.vectorize
def compute_rest(anchor_spring : AnchorSpring, details):
    x = db.x(details.node, anchor_spring.node_IDs[0])
    anchor_spring.rest_length = np.float64(math2D.distance(anchor_spring.kinematic_component_pos, x))

@generate.vectorize
def compute_forces(anchor_spring : AnchorSpring, details):
    x, v = db.xv(details.node, anchor_spring.node_IDs[0])
    kinematic_vel = np.zeros(2)
    target_pos = anchor_spring.kinematic_component_pos
    force = spring_lib.spring_stretch_force(x, target_pos, anchor_spring.rest_length, anchor_spring.stiffness)
    force += spring_lib.spring_damping_force(x, target_pos, v, kinematic_vel, anchor_spring.damping)
    anchor_spring.f = force

@generate.vectorize
def compute_force_jacobians(anchor_spring : AnchorSpring, details):
    x, v = db.xv(details.node, anchor_spring.node_IDs[0])
    kinematic_vel = np.zeros(2)
    target_pos = anchor_spring.kinematic_component_pos
    dfdx = spring_lib.spring_stretch_jacobian(x, target_pos, anchor_spring.rest_length, anchor_spring.stiffness)
    dfdv = spring_lib.spring_damping_jacobian(x, target_pos, v, kinematic_vel, anchor_spring.damping)
    anchor_spring.dfdx[0][0] = dfdx
    anchor_spring.dfdv[0][0] = dfdv
