"""
@author: Vincent Bonnet
@description : Bending Constraint for the implicit solver
"""

from constraints.base import Base
import core.differentiation as diff
import core.math_2d as math2D
from numba import njit
from math import atan2, sqrt
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
        force0 = elasticBendingForce0(x0, x1, x2, self.rest_angle, self.stiffness)
        force1 = elasticBendingForce1(x0, x1, x2, self.rest_angle, self.stiffness)
        force2 = elasticBendingForce2(x0, x1, x2, self.rest_angle, self.stiffness)
        # Set forces
        self.f[0] = force0
        self.f[1] = force1
        self.f[2] = force2

    def compute_jacobians(self, scene):
        x0, x1, x2, v0, v1, v2 = self.get_states(scene)
        # Numerical jacobians (Aka Hessian of the energy)
        dfdx00 = diff.numerical_hessian(elasticBendingEnergy, 0, 0, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        dfdx11 = diff.numerical_hessian(elasticBendingEnergy, 1, 1, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        dfdx22 = diff.numerical_hessian(elasticBendingEnergy, 2, 2, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        dfdx01 = diff.numerical_hessian(elasticBendingEnergy, 0, 1, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        dfdx02 = diff.numerical_hessian(elasticBendingEnergy, 0, 2, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        dfdx12 = diff.numerical_hessian(elasticBendingEnergy, 1, 2, x0, x1, x2, self.rest_angle, self.stiffness) * -1.0
        # Numerical jacobians from forces
        # TODO
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

def elasticBendingForce0(x0, x1, x2, rest_angle, stiffness):
    force = np.zeros(2)
    px0 = x0[0]
    py0 = x0[1]
    px1 = x1[0]
    py1 = x1[1]
    px2 = x2[0]
    py2 = x2[1]
    force[0] = stiffness*(rest_angle - atan2((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1), (px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)))*(0.25*(px0 - px1)*(rest_angle - atan2((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1), (px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)))*(((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2))**2 + ((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1))**2) + 0.5*((px1 - px2)*((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1)) - (py1 - py2)*((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)))*sqrt((px0 - px1)**2 + (py0 - py1)**2)*(sqrt((px0 - px1)**2 + (py0 - py1)**2) + sqrt((px1 - px2)**2 + (py1 - py2)**2)))/(sqrt((px0 - px1)**2 + (py0 - py1)**2)*(((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2))**2 + ((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1))**2))
    force[1] = stiffness*(rest_angle - atan2((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1), (px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)))*(0.25*(py0 - py1)*(rest_angle - atan2((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1), (px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)))*(((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2))**2 + ((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1))**2) + 0.5*((px1 - px2)*((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)) + (py1 - py2)*((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1)))*sqrt((px0 - px1)**2 + (py0 - py1)**2)*(sqrt((px0 - px1)**2 + (py0 - py1)**2) + sqrt((px1 - px2)**2 + (py1 - py2)**2)))/(sqrt((px0 - px1)**2 + (py0 - py1)**2)*(((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2))**2 + ((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1))**2))
    return force * -1.0

def elasticBendingForce2(x0, x1, x2, rest_angle, stiffness):
    force = np.zeros(2)
    px0 = x0[0]
    py0 = x0[1]
    px1 = x1[0]
    py1 = x1[1]
    px2 = x2[0]
    py2 = x2[1]
    force[0] = -stiffness*(rest_angle - atan2((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1), (px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)))*(0.25*(px1 - px2)*(rest_angle - atan2((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1), (px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)))*(((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2))**2 + ((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1))**2) + 0.5*((px0 - px1)*((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1)) + (py0 - py1)*((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)))*sqrt((px1 - px2)**2 + (py1 - py2)**2)*(sqrt((px0 - px1)**2 + (py0 - py1)**2) + sqrt((px1 - px2)**2 + (py1 - py2)**2)))/(sqrt((px1 - px2)**2 + (py1 - py2)**2)*(((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2))**2 + ((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1))**2))
    force[1] = stiffness*(rest_angle - atan2((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1), (px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)))*(-0.25*(py1 - py2)*(rest_angle - atan2((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1), (px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)))*(((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2))**2 + ((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1))**2) + 0.5*((px0 - px1)*((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2)) - (py0 - py1)*((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1)))*sqrt((px1 - px2)**2 + (py1 - py2)**2)*(sqrt((px0 - px1)**2 + (py0 - py1)**2) + sqrt((px1 - px2)**2 + (py1 - py2)**2)))/(sqrt((px1 - px2)**2 + (py1 - py2)**2)*(((px0 - px1)*(px1 - px2) + (py0 - py1)*(py1 - py2))**2 + ((px0 - px1)*(py1 - py2) - (px1 - px2)*(py0 - py1))**2))
    return force * -1.0

def elasticBendingForce1(x0, x1, x2, rest_angle, stiffness):
    f0 = elasticBendingForce0(x0, x1, x2, rest_angle, stiffness)
    f2 = elasticBendingForce2(x0, x1, x2, rest_angle, stiffness)
    force = np.zeros(2)
    return (force - f0 - f2)





