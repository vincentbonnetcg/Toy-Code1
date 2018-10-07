"""
@author: Vincent Bonnet
@description : Bending Constraint for the implicit solver
"""

import numpy as np
import constraints.differentiation as diff
from constraints.base import Base
import constraints.fastMath as fastMath

class Bending(Base):
    '''
    Describes a 2D bending constraint between three particles
    '''
    def __init__(self, stiffness, damping, dynamics, particleIds):
        Base.__init__(self, stiffness, damping, dynamics, particleIds)
        # Constraint three points
        #  x0 -- x1 -- x2
        x0 = dynamics[0].x[particleIds[0]]
        x1 = dynamics[1].x[particleIds[1]]
        x2 = dynamics[2].x[particleIds[2]]
        self.restCurvature = computeCurvature(x0, x1, x2)

    def getStates(self, scene):
        dynamic0 = scene.dynamics[self.dynamicIndices[0]]
        dynamic1 = scene.dynamics[self.dynamicIndices[1]]
        dynamic2 = scene.dynamics[self.dynamicIndices[2]]
        x0 = dynamic0.x[self.localIds[0]]
        x1 = dynamic1.x[self.localIds[1]]
        x2 = dynamic2.x[self.localIds[2]]
        v0 = dynamic0.v[self.localIds[0]]
        v1 = dynamic1.v[self.localIds[1]]
        v2 = dynamic2.v[self.localIds[1]]
        return (x0, x1, x2, v0, v1, v2)

    def computeForces(self, scene):
        x0, x1, x2, v0, v1, v2 = self.getStates(scene)
        # Numerical forces
        force0 = diff.numericalJacobian(elasticBendingEnergy, 0, x0, x1, x2, self.restCurvature, self.stiffness) * -1.0
        force1 = diff.numericalJacobian(elasticBendingEnergy, 1, x0, x1, x2, self.restCurvature, self.stiffness) * -1.0
        force2 = diff.numericalJacobian(elasticBendingEnergy, 2, x0, x1, x2, self.restCurvature, self.stiffness) * -1.0
        # Analytic forces
        # TODO
        # Set forces
        self.f[0] = force0
        self.f[1] = force1
        self.f[2] = force2

    def computeJacobians(self, scene):
        x0, x1, x2, v0, v1, v2 = self.getStates(scene)
        # Numerical jacobians (Aka Hessian of the energy)
        dfdx00 = diff.numericalHessian(elasticBendingEnergy, 0, 0, x0, x1, x2, self.restCurvature, self.stiffness) * -1.0
        dfdx11 = diff.numericalHessian(elasticBendingEnergy, 1, 1, x0, x1, x2, self.restCurvature, self.stiffness) * -1.0
        dfdx22 = diff.numericalHessian(elasticBendingEnergy, 2, 2, x0, x1, x2, self.restCurvature, self.stiffness) * -1.0
        dfdx01 = diff.numericalHessian(elasticBendingEnergy, 0, 1, x0, x1, x2, self.restCurvature, self.stiffness) * -1.0
        dfdx02 = diff.numericalHessian(elasticBendingEnergy, 0, 2, x0, x1, x2, self.restCurvature, self.stiffness) * -1.0
        dfdx12 = diff.numericalHessian(elasticBendingEnergy, 1, 2, x0, x1, x2, self.restCurvature, self.stiffness) * -1.0
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
def computeCurvature(x0, x1, x2):
    '''
    Connect three points :
      x1
      /\
     /  \
    x0  x2
    Compute the curvature : |dT/ds| where T is the tangent and s the surface
    with
    t01 = x1 - x0 and t12 = x2 - x1
    mid01 = (x0 + x1) * 0.5
    mid12 = (x1 + x2) * 0.5
    discrete curvate formula 1: |t12 - t01| / |mid12 - mid01|
    discrete curvate formula 2: angle(t12,t01) / |mid12 - mid01|
    '''
    t01 = x1 - x0
    t01 /= fastMath.norm(t01)
    t12 = x2 - x1
    t12 /= fastMath.norm(t12)
    #mid01 = (x0 + x1) * 0.5
    #mid12 = (x1 + x2) * 0.5
    # Discrete curvature - poor (1)
    #curvature = fastMath.norm(t12 - t01) #/ fastMath.norm(mid12 - mid01)
    # Discrete curvature - accurate (2)
    det = t01[0]*t12[1] - t01[1]*t12[0]      # determinant
    dot = t01[0]*t12[0] + t01[1]*t12[1]      # dot product
    angle = np.math.atan2(det,dot)  # atan2 return range [-pi, pi]
    # TOFIX : instability to fix
    curvature = angle # / fastMath.norm(mid12 - mid01)
    return curvature

def elasticBendingEnergy(x0, x1, x2, restCurvature, stiffness):
    curvature = computeCurvature(x0, x1, x2)
    length = fastMath.norm(x1 - x0) + fastMath.norm(x2 - x1)
    return 0.5 * stiffness * ((curvature - restCurvature) * (curvature - restCurvature)) * length
