"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np

class Base:
    '''
    Describes the constraint base
    '''
    def __init__(self, stiffness, damping, dynamics, particles_ids):
        N = len(particles_ids) # number of particles involved in the constraint
        self.stiffness = stiffness
        self.damping = damping
        self.f = np.zeros((N, 2))
        # Particle identifications
        self.localIds = np.copy(particles_ids) # local particle indices
        self.dynamicIndices = np.zeros(N, dtype=int) # indices of the dynamic objects
        self.globalIds = np.zeros(N, dtype=int)
        for i in range(N):
            self.dynamicIndices[i] = dynamics[i].index
            self.globalIds[i] = self.localIds[i] + dynamics[i].global_offset
        # Precomputed jacobians.
        # NxN matrix where each element is a 2x2 submatrix
        self.dfdx = np.zeros((N, N, 2, 2))
        self.dfdv = np.zeros((N, N, 2, 2))

    def applyForces(self, scene):
        for i in range(len(self.localIds)):
            dynamic = scene.dynamics[self.dynamicIndices[i]]
            dynamic.f[self.localIds[i]] += self.f[i]

    def computeForces(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeForces'")

    def computeJacobians(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeJacobians'")

    def getJacobianDx(self, fi, xj):
        return self.dfdx[fi][xj]

    def getJacobianDv(self, fi, xj):
        return self.dfdv[fi][xj]


