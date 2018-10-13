"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np
import constraints.differentiation as diff
from constraints.base import Base

class Area(Base):
    '''
    Describes a 2D area constraint between three particles
    '''
    def __init__(self, stiffness, damping, dynamics, particleIds):
        Base.__init__(self, stiffness, damping, dynamics, particleIds)
        x0 = dynamics[0].x[particleIds[0]]
        x1 = dynamics[1].x[particleIds[1]]
        x2 = dynamics[2].x[particleIds[2]]
        v01 = np.subtract(x1, x0)
        v02 = np.subtract(x2, x0)
        self.restArea = np.abs(np.cross(v01, v02)) * 0.5

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
        force0 = diff.numerical_jacobian(elasticAreaEnergy, 0, x0, x1, x2, self.restArea, self.stiffness) * -1.0
        force1 = diff.numerical_jacobian(elasticAreaEnergy, 1, x0, x1, x2, self.restArea, self.stiffness) * -1.0
        force2 = diff.numerical_jacobian(elasticAreaEnergy, 2, x0, x1, x2, self.restArea, self.stiffness) * -1.0
        # Analytic forces
        # TODO
        # Set forces
        self.f[0] = force0
        self.f[1] = force1
        self.f[2] = force2

    def computeJacobians(self, scene):
        x0, x1, x2, v0, v1, v2 = self.getStates(scene)
        # Numerical jacobians (Aka Hessian of the energy)
        dfdx00 = diff.numerical_hessian(elasticAreaEnergy, 0, 0, x0, x1, x2, self.restArea, self.stiffness) * -1.0
        dfdx11 = diff.numerical_hessian(elasticAreaEnergy, 1, 1, x0, x1, x2, self.restArea, self.stiffness) * -1.0
        dfdx22 = diff.numerical_hessian(elasticAreaEnergy, 2, 2, x0, x1, x2, self.restArea, self.stiffness) * -1.0
        dfdx01 = diff.numerical_hessian(elasticAreaEnergy, 0, 1, x0, x1, x2, self.restArea, self.stiffness) * -1.0
        dfdx02 = diff.numerical_hessian(elasticAreaEnergy, 0, 2, x0, x1, x2, self.restArea, self.stiffness) * -1.0
        dfdx12 = diff.numerical_hessian(elasticAreaEnergy, 1, 2, x0, x1, x2, self.restArea, self.stiffness) * -1.0
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
def elasticAreaEnergy(x0, x1, x2, restArea, stiffness):
    u = x1 - x0 # np.subtract(x1, x0)
    v = x2 - x0 # np.subtract(x2, x0)
    #area = np.abs(np.cross(u, v)) * 0.5 # expensive operation => replaced with line below
    area = np.abs(u[0]*v[1]-v[0]*u[1]) * 0.5
    return 0.5 * stiffness * ((area - restArea) * (area - restArea))



