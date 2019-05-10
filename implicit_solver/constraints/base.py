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
    def __init__(self, scene, stiffness, damping, node_ids):
        N = len(node_ids) # number of nodes involved in the constraint
        self.stiffness = stiffness
        self.damping = damping
        self.f = np.zeros((N, 2)) # forces
        self.n_ids = np.copy(node_ids) # node ids
        # Precomputed jacobians.
        # NxN matrix where each element is a 2x2 submatrix
        self.dfdx = np.zeros((N, N, 2, 2))
        self.dfdv = np.zeros((N, N, 2, 2))

    def apply_forces(self, scene):
        for node_id in range(len(self.n_ids)):
            scene.node_add_f(self.n_ids[node_id], self.f[node_id])

    def compute_forces(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'compute_forces'")

    def compute_jacobians(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'compute_jacobians'")

    def jacobian_dx(self, fi, xj):
        return self.dfdx[fi][xj]

    def jacobian_dv(self, fi, xj):
        return self.dfdv[fi][xj]

    @staticmethod
    def specify_data_block(data_block : core.DataBlock):
        print("BUILD BASE DATA")
        # TODO
