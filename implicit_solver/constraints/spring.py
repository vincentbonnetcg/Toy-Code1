"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np
from constraints.base import Base
import core.math_2d as math2D
import core.differentiation as diff
from numba import njit

class AnchorSpring(Base):
    '''
    Describes a 2D spring constraint between a particle and point
    '''
    def __init__(self, scene, stiffness, damping, node_id, kinematic, point_params):
        Base.__init__(self, scene, stiffness, damping, [node_id])
        target_pos = kinematic.get_point_from_parametric_value(point_params)
        x, v = scene.n_state(node_id)
        self.rest_length = math2D.distance(target_pos, x)
        self.point_params = point_params
        self.kinematic_index = kinematic.index
        self.kinematic_vel = np.zeros(2) # No velocity associated to kinematic object

    def get_states(self, scene):
        kinematic = scene.kinematics[self.kinematic_index]
        x, v = scene.n_state(self.n_ids[0])
        return (kinematic, x, v)

    def compute_forces(self, scene):
        kinematic, x, v = self.get_states(scene)
        target_pos = kinematic.get_point_from_parametric_value(self.point_params)
        # Numerical forces
        #force = diff.numerical_jacobian(elasticSpringEnergy, 0, x, target_pos, self.rest_length, self.stiffness) * -1.0
        # Analytic forces
        force = springStretchForce(x, target_pos, self.rest_length, self.stiffness)
        force += springDampingForce(x, target_pos, v, self.kinematic_vel, self.damping)
        # Set forces
        self.f[0] = force

    def compute_jacobians(self, scene):
        kinematic, x, v = self.get_states(scene)
        target_pos = kinematic.get_point_from_parametric_value(self.point_params)
        # Numerical jacobians
        #dfdx = diff.numerical_jacobian(springStretchForce, 0, x, target_pos, self.rest_length, self.stiffness)
        #dfdv = diff.numerical_jacobian(springDampingForce, 2, x, target_pos, v, (0,0), self.damping)
        # Analytic jacobians
        dfdx = springStretchJacobian(x, target_pos, self.rest_length, self.stiffness)
        dfdv = springDampingJacobian(x, target_pos, v, self.kinematic_vel, self.damping)
        # Set jacobians
        self.dfdx[0][0] = dfdx
        self.dfdv[0][0] = dfdv

class Spring(Base):
    '''
    Describes a 2D spring constraint between two particles
    '''
    def __init__(self, scene, stiffness, damping, node_ids):
        Base.__init__(self, scene, stiffness, damping, node_ids)
        x0, v0 = scene.n_state(self.n_ids[0])
        x1, v1 = scene.n_state(self.n_ids[1])
        self.rest_length = math2D.distance(x0, x1)

    def get_states(self, scene):
        x0, v0 = scene.n_state(self.n_ids[0])
        x1, v1 = scene.n_state(self.n_ids[1])
        return (x0, x1, v0, v1)

    def compute_forces(self, scene):
        x0, x1, v0, v1 = self.get_states(scene)
        # Numerical forces
        #force = diff.numerical_jacobian(elasticSpringEnergy, 0, x0, x1, self.rest_length, self.stiffness) * -1.0
        # Analytic forces
        force = springStretchForce(x0, x1, self.rest_length, self.stiffness)
        force += springDampingForce(x0, x1, v0, v1, self.damping)
        # Set forces
        self.f[0] = force
        self.f[1] = force * -1

    def compute_jacobians(self, scene):
        x0, x1, v0, v1 = self.get_states(scene)
        # Numerical jacobians
        #dfdx = diff.numerical_jacobian(springStretchForce, 0, x0, x1, self.rest_length, self.stiffness)
        #dfdv = diff.numerical_jacobian(springDampingForce, 2, x0, x1, v0, v1, self.damping)
        # Analytic jacobians
        dfdx = springStretchJacobian(x0, x1, self.rest_length, self.stiffness)
        dfdv = springDampingJacobian(x0, x1, v0, v1, self.damping)
        # Set jacobians
        self.dfdx[0][0] = self.dfdx[1][1] = dfdx
        self.dfdx[0][1] = self.dfdx[1][0] = dfdx * -1
        self.dfdv[0][0] = self.dfdv[1][1] = dfdv
        self.dfdv[0][1] = self.dfdv[1][0] = dfdv * -1

'''
 Utility Functions
'''
@njit
def springStretchJacobian(x0, x1, rest, stiffness):
    direction = x0 - x1
    stretch = math2D.norm(direction)
    I = np.identity(2)
    if not math2D.is_close(stretch, 0.0):
        direction /= stretch
        A = np.outer(direction, direction)
        return -1.0 * stiffness * ((1 - (rest / stretch)) * (I - A) + A)

    return -1.0 * stiffness * I

@njit
def springDampingJacobian(x0, x1, v0, v1, damping):
    jacobian = np.zeros(shape=(2, 2))
    direction = x1 - x0
    stretch = math2D.norm(direction)
    if not math2D.is_close(stretch, 0.0):
        direction /= stretch
        A = np.outer(direction, direction)
        jacobian = -1.0 * damping * A

    return jacobian

@njit
def springStretchForce(x0, x1, rest, stiffness):
    direction = x1 - x0
    stretch = math2D.norm(direction)
    if not math2D.is_close(stretch, 0.0):
        direction /= stretch
    return direction * ((stretch - rest) * stiffness)

@njit
def springDampingForce(x0, x1, v0, v1, damping):
    direction = x1 - x0
    stretch = math2D.norm(direction)
    if not math2D.is_close(stretch, 0.0):
        direction /= stretch
    relativeVelocity = v1 - v0
    return direction * (np.dot(relativeVelocity, direction) * damping)

@njit
def elasticSpringEnergy(x0, x1, rest, stiffness):
    stretch = math2D.distance(x0, x1)
    return 0.5 * stiffness * ((stretch - rest)**2)
