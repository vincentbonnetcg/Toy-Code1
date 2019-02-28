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
        self.particles_ids = np.copy(particles_ids) # local particle indices
        self.dynamic_ids = np.zeros(N, dtype=int) # indices of the dynamic objects
        self.global_particle_ids = np.zeros(N, dtype=int) # global particle indices
        for i in range(N):
            self.dynamic_ids[i] = dynamics[i].index
            self.global_particle_ids[i] = self.particles_ids[i] + dynamics[i].global_offset
        # Precomputed jacobians.
        # NxN matrix where each element is a 2x2 submatrix
        self.dfdx = np.zeros((N, N, 2, 2))
        self.dfdv = np.zeros((N, N, 2, 2))

    def applyForces(self, scene):
        for i in range(len(self.particles_ids)):
            dynamic = scene.dynamics[self.dynamic_ids[i]]
            dynamic.f[self.particles_ids[i]] += self.f[i]

    def computeForces(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeForces'")

    def computeJacobians(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeJacobians'")

    def getJacobianDx(self, fi, xj):
        return self.dfdx[fi][xj]

    def getJacobianDv(self, fi, xj):
        return self.dfdv[fi][xj]


