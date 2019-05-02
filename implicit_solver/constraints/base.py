"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np
import core

class Base:
    '''
    Describes the constraint base
    '''
    def __init__(self, stiffness, damping, dynamics, particles_ids):
        N = len(particles_ids) # number of particles involved in the constraint
        self.stiffness = stiffness
        self.damping = damping
        self.f = np.zeros((N, 2))
        self.n_ids = np.zeros((N, 2), dtype=int)  # node identifiers
        self.global_particle_ids = np.zeros(N, dtype=int) # global particle indices
        for i in range(N):
            self.global_particle_ids[i] = particles_ids[i] + dynamics[i].global_offset
            self.n_ids[i][0] = dynamics[i].index
            self.n_ids[i][1] = particles_ids[i]

        # Precomputed jacobians.
        # NxN matrix where each element is a 2x2 submatrix
        self.dfdx = np.zeros((N, N, 2, 2))
        self.dfdv = np.zeros((N, N, 2, 2))

    def apply_forces(self, scene):
        for node_id in range(len(self.n_ids)):
            scene.n_add_f(self.n_ids[node_id], self.f[node_id])

    def compute_forces(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'compute_forces'")

    def compute_jacobians(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'compute_jacobians'")

    def getJacobianDx(self, fi, xj):
        return self.dfdx[fi][xj]

    def getJacobianDv(self, fi, xj):
        return self.dfdv[fi][xj]

    @staticmethod
    def specify_data_block(data_block : core.DataBlock):
        print("BUILD BASE DATA")
        # TODO


