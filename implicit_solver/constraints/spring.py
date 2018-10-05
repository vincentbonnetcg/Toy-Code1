"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np
from constraints.base import Base
import constraints.fastMath as fastMath

class AnchorSpring(Base):
    '''
    Describes a 2D spring constraint between a particle and point
    '''
    def __init__(self, stiffness, damping, dynamic, particleId, kinematic, pointParams):
        Base.__init__(self, stiffness, damping, [dynamic], [particleId])
        targetPos = kinematic.getPointFromParametricValues(pointParams)
        self.restLength = np.linalg.norm(targetPos - dynamic.x[self.localIds[0]])
        self.pointParams = pointParams
        self.kinematicIndex = kinematic.index

    def getStates(self, scene):
        kinematic = scene.kinematics[self.kinematicIndex]
        dynamic = scene.dynamics[self.dynamicIndices[0]]
        x = dynamic.x[self.localIds[0]]
        v = dynamic.v[self.localIds[0]]
        return (kinematic, x, v)

    def computeForces(self, scene):
        kinematic, x, v = self.getStates(scene)
        targetPos = kinematic.getPointFromParametricValues(self.pointParams)
        # Numerical forces
        # f = -de/dx : the force is f,  e is the energy and x is the position
        #force = diff.numericalJacobian(elasticSpringEnergy, 0, x, targetPos, self.restLength, self.stiffness) * -1.0
        # Analytic forces
        force = springStretchForce(x, targetPos, self.restLength, self.stiffness)
        force += springDampingForce(x, targetPos, v, (0, 0), self.damping)
        # Set forces
        self.f[0] = force

    def computeJacobians(self, scene):
        kinematic, x, v = self.getStates(scene)
        targetPos = kinematic.getPointFromParametricValues(self.pointParams)
        # Numerical jacobians
        #dfdx = diff.numericalJacobian(springStretchForce, 0, x, targetPos, self.restLength, self.stiffness)
        #dfdv = diff.numericalJacobian(springDampingForce, 2, x, targetPos, v, (0,0), self.damping)
        # Analytic jacobians
        dfdx = springStretchJacobian(x, targetPos, self.restLength, self.stiffness)
        dfdv = springDampingJacobian(x, targetPos, v, (0, 0), self.damping)
        # Set jacobians
        self.dfdx[0][0] = dfdx
        self.dfdv[0][0] = dfdv

class Spring(Base):
    '''
    Describes a 2D spring constraint between two particles
    '''
    def __init__(self, stiffness, damping, dynamics, particleIds):
        Base.__init__(self, stiffness, damping, dynamics, particleIds)
        self.restLength = np.linalg.norm(dynamics[0].x[particleIds[0]] - dynamics[1].x[particleIds[1]])

    def getStates(self, scene):
        dynamic0 = scene.dynamics[self.dynamicIndices[0]]
        dynamic1 = scene.dynamics[self.dynamicIndices[1]]
        x0 = dynamic0.x[self.localIds[0]]
        x1 = dynamic1.x[self.localIds[1]]
        v0 = dynamic0.v[self.localIds[0]]
        v1 = dynamic1.v[self.localIds[1]]
        return (x0, x1, v0, v1)

    def computeForces(self, scene):
        x0, x1, v0, v1 = self.getStates(scene)
        # Numerical forces
        # f = -de/dx : the force is f,  e is the energy and x is the position
        #force = diff.numericalJacobian(elasticSpringEnergy, 0, x0, x1, self.restLength, self.stiffness) * -1.0
        # Analytic forces
        force = springStretchForce(x0, x1, self.restLength, self.stiffness)
        force += springDampingForce(x0, x1, v0, v1, self.damping)
        # Set forces
        self.f[0] = force
        self.f[1] = force * -1

    def computeJacobians(self, scene):
        x0, x1, v0, v1 = self.getStates(scene)
        # Numerical jacobians
        #dfdx = diff.numericalJacobian(springStretchForce, 0, x0, x1, self.restLength, self.stiffness)
        #dfdv = diff.numericalJacobian(springDampingForce, 2, x0, x1, v0, v1, self.damping)
        # Analytic jacobians
        dfdx = springStretchJacobian(x0, x1, self.restLength, self.stiffness)
        dfdv = springDampingJacobian(x0, x1, v0, v1, self.damping)
        # Set jacobians
        self.dfdx[0][0] = dfdx
        self.dfdx[1][1] = dfdx
        self.dfdx[0][1] = dfdx * -1
        self.dfdx[1][0] = dfdx * -1

        self.dfdv[0][0] = dfdv
        self.dfdv[1][1] = dfdv
        self.dfdv[0][1] = dfdv * -1
        self.dfdv[1][0] = dfdv * -1

'''
 Utility Functions
'''
# direction = normalized(x0-x1)
# stretch = norm(direction)
# A = outerProduct(direction, direction)
# I = identity matrix
# J =  -stiffness * [(1 - rest / stretch)(I - A) + A]
def springStretchJacobian(x0, x1, rest, stiffness):
    jacobian = np.zeros(shape=(2, 2))
    direction = x0 - x1
    stretch = fastMath.norm(direction)
    I = np.identity(2)
    if not np.isclose(stretch, 0.0):
        direction /= stretch
        A = np.outer(direction, direction)
        jacobian = -1.0 * stiffness * ((1 - (rest / stretch)) * (I - A) + A)
    else:
        jacobian = -1.0 * stiffness * I

    return jacobian

def springDampingJacobian(x0, x1, v0, v1, damping):
    jacobian = np.zeros(shape=(2, 2))
    direction = x1 - x0
    stretch = fastMath.norm(direction)
    if not np.isclose(stretch, 0.0):
        direction /= stretch
        A = np.outer(direction, direction)
        jacobian = -1.0 *damping * A

    return jacobian

def springStretchForce(x0, x1, rest, stiffness):
    direction = x1 - x0
    stretch = fastMath.norm(direction)
    if not np.isclose(stretch, 0.0):
        direction /= stretch
    return direction * ((stretch - rest) * stiffness)

def springDampingForce(x0, x1, v0, v1, damping):
    direction = x1 - x0
    stretch = fastMath.norm(direction)
    if not np.isclose(stretch, 0.0):
        direction /= stretch
    relativeVelocity = v1 - v0
    return direction * (np.dot(relativeVelocity, direction) * damping)

def elasticSpringEnergy(x0, x1, rest, stiffness):
    direction = x1 - x0
    stretch = fastMath.norm(direction)
    return 0.5 * stiffness * ((stretch - rest) * (stretch - rest))