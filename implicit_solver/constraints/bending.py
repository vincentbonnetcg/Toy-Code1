"""
@author: Vincent Bonnet
@description : Bending Constraint for the implicit solver
"""

from constraints.base import Base
import core.math_2d as math2D
from numba import njit
import math
import numpy as np

class Bending(Base):
    '''
    Describes a 2D bending constraint of a thin inextensible wire
    between three particles.
    This bending is NOT the proper bending formulation and uses angle instead of curvature
    Some instabilities when using the curvature => Need to investigate
    '''
    def __init__(self, stiffness, damping, dynamics, particle_ids):
        Base.__init__(self, stiffness, damping, dynamics, particle_ids)
        # Constraint three points
        #  x0 -- x1 -- x2
        x0 = dynamics[0].x[particle_ids[0]]
        x1 = dynamics[1].x[particle_ids[1]]
        x2 = dynamics[2].x[particle_ids[2]]
        self.rest_angle = math2D.angle(x0, x1, x2)

    def get_states(self, scene):
        dynamic0 = scene.dynamics[self.dynamic_ids[0]]
        dynamic1 = scene.dynamics[self.dynamic_ids[1]]
        dynamic2 = scene.dynamics[self.dynamic_ids[2]]
        x0 = dynamic0.x[self.particles_ids[0]]
        x1 = dynamic1.x[self.particles_ids[1]]
        x2 = dynamic2.x[self.particles_ids[2]]
        v0 = dynamic0.v[self.particles_ids[0]]
        v1 = dynamic1.v[self.particles_ids[1]]
        v2 = dynamic2.v[self.particles_ids[1]]
        return (x0, x1, x2, v0, v1, v2)

    def compute_forces(self, scene):
        x0, x1, x2, v0, v1, v2 = self.get_states(scene)
        # Numerical forces
        #force0 = diff.numerical_jacobian(elasticBendingEnergy, 0, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        #force1 = diff.numerical_jacobian(elasticBendingEnergy, 1, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        #force2 = diff.numerical_jacobian(elasticBendingEnergy, 2, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        # Analytic forces
        force0, force1, force2 = elasticBendingForces(x0, x1, x2, self.rest_angle, self.stiffness, [True, True, True])
        # Set forces
        self.f[0] = force0
        self.f[1] = force1
        self.f[2] = force2

    def compute_jacobians(self, scene):
        x0, x1, x2, v0, v1, v2 = self.get_states(scene)
        # Numerical jacobians (Aka Hessian of the energy)
        #dfdx00 = diff.numerical_hessian(elasticBendingEnergy, 0, 0, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        #dfdx11 = diff.numerical_hessian(elasticBendingEnergy, 1, 1, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        #dfdx22 = diff.numerical_hessian(elasticBendingEnergy, 2, 2, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        #dfdx01 = diff.numerical_hessian(elasticBendingEnergy, 0, 1, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        #dfdx02 = diff.numerical_hessian(elasticBendingEnergy, 0, 2, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        #dfdx12 = diff.numerical_hessian(elasticBendingEnergy, 1, 2, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        # Numerical jacobians from forces
        jacobians = elasticBendingNumericalJacobians(x0, x1, x2, self.rest_angle, self.stiffness)
        dfdx00 = jacobians[0]
        dfdx11 = jacobians[1]
        dfdx22 = jacobians[2]
        dfdx01 = jacobians[3]
        dfdx02 = jacobians[4]
        dfdx12 = jacobians[5]
        # Analytic jacobians
        # TODO
        # Set jacobians
        self.dfdx[0][0] = dfdx00
        self.dfdx[1][1] = dfdx11
        self.dfdx[2][2] = dfdx22
        self.dfdx[0][1] = self.dfdx[1][0] = dfdx01
        self.dfdx[0][2] = self.dfdx[2][0] = dfdx02
        self.dfdx[1][2] = self.dfdx[2][1] = dfdx12

'''
 Utility Functions
'''
@njit
def elasticBendingEnergy(x0, x1, x2, rest_angle, stiffness):
    angle = math2D.angle(x0, x1, x2)
    arc_length = (math2D.norm(x1 - x0) + math2D.norm(x2 - x1)) * 0.5
    return 0.5 * stiffness * ((angle - rest_angle)**2) * arc_length

@njit
def elasticBendingForces(x0, x1, x2, rest_angle, stiffness, enable_force = [True, True, True]):
    forces = np.zeros((3, 2))

    u = x0 - x1
    v = x1 - x2
    det = u[0]*v[1] - v[0]*u[1]
    dot = u[0]*v[0] + u[1]*v[1]

    norm_u = math.sqrt(u[0]**2 + u[1]**2)
    norm_v = math.sqrt(v[0]**2 + v[1]**2)

    diff_angle = rest_angle - math.atan2(det, dot)

    if enable_force[0] or enable_force[1]:
        forces[0][0] = v[0]*det - v[1]*dot
        forces[0][1] = v[0]*dot + v[1]*det

        forces[0] *= 0.5*norm_u*(norm_u + norm_v)
        forces[0] += 0.25*u*diff_angle*(dot**2 + det**2)

        forces[0] /= norm_u*(dot**2 + det**2)
        forces[0] *= stiffness*diff_angle*-1.0

    if enable_force[2] or enable_force[1]:
        forces[2][0] = -(u[0]*det + u[1]*dot)
        forces[2][1] = u[0]*dot - u[1]*det

        forces[2] *= 0.5 * norm_v*(norm_u + norm_v)
        forces[2] += -0.25 * v * diff_angle * (dot**2 + det**2)

        forces[2] /= norm_v*(dot**2 + det**2)
        forces[2] *= stiffness*diff_angle*-1.0

    if enable_force[1]:
        forces[1] -= (forces[0] + forces[2])

    return forces

@njit
def elasticBendingNumericalJacobians(x0, x1, x2, rest_angle, stiffness):
    '''
    Returns the six jacobians matrices in the following order
    df0dx0, df1dx1, df2dx2, df0dx1, df0dx2, df1dx2
    dfdx01 is the derivative of f0 relative to x1
    etc.
    '''
    jacobians = np.zeros(shape=(6, 2, 2))
    STENCIL_SIZE = 1e-6

    # derivate of f0 relative to x0
    x_cids = [0, 1] # x component id means x[0], x[1]
    f_ids = [0, 0] # is the force id
    for gradient_id in range(2):
        x_cid = x_cids[gradient_id]
        f_id = f_ids[gradient_id]
        x0_ = np.copy(x0)
        x0_[x_cid] = x0[x_cid]+STENCIL_SIZE
        forces = elasticBendingForces(x0_, x1, x2, rest_angle, stiffness, [True, False, False])
        gradient = forces[f_id]
        x0_[x_cid] = x0[x_cid]-STENCIL_SIZE
        forces = elasticBendingForces(x0_, x1, x2, rest_angle, stiffness, [True, False, False])
        gradient -= forces[f_id]
        gradient /= (2.0 * STENCIL_SIZE)
        jacobians[0, 0:2, x_cid] = gradient

    # derivate of f0, f1 relative to x1
    x_cids = [0, 1] # x component id means x[0], x[1]
    for gradient_id in range(2):
        x_cid = x_cids[gradient_id]
        x1_ = np.copy(x1)
        x1_[x_cid] = x1[x_cid]+STENCIL_SIZE
        forces = elasticBendingForces(x0, x1_, x2, rest_angle, stiffness, [True, True, False])
        grad_f0_x1 = forces[0]
        grad_f1_x1 = forces[1]
        x1_[x_cid] = x1[x_cid]-STENCIL_SIZE
        forces = elasticBendingForces(x0, x1_, x2, rest_angle, stiffness, [True, True, False])
        grad_f0_x1 -= forces[0]
        grad_f1_x1 -= forces[1]
        jacobians[1, 0:2, x_cid] = grad_f1_x1 / (2.0 * STENCIL_SIZE)
        jacobians[3, 0:2, x_cid] = grad_f0_x1 / (2.0 * STENCIL_SIZE)

    # derivate of f0, f1, f2 relative to x2
    x_cids = [0, 1] # x component id means x[0], x[1]
    for gradient_id in range(2):
        x_cid = x_cids[gradient_id]
        x2_ = np.copy(x2)
        x2_[x_cid] = x2[x_cid]+STENCIL_SIZE
        forces = elasticBendingForces(x0, x1, x2_, rest_angle, stiffness, [True, True, True])
        grad_f0_x2 = forces[0]
        grad_f1_x2 = forces[1]
        grad_f2_x2 = forces[2]
        x2_[x_cid] = x2[x_cid]-STENCIL_SIZE
        forces = elasticBendingForces(x0, x1, x2_, rest_angle, stiffness, [True, True, True])
        grad_f0_x2 -= forces[0]
        grad_f1_x2 -= forces[1]
        grad_f2_x2 -= forces[2]
        jacobians[4, 0:2, x_cid] = grad_f0_x2 / (2.0 * STENCIL_SIZE)
        jacobians[5, 0:2, x_cid] = grad_f1_x2 / (2.0 * STENCIL_SIZE)
        jacobians[2, 0:2, x_cid] = grad_f2_x2 / (2.0 * STENCIL_SIZE)

    return jacobians
