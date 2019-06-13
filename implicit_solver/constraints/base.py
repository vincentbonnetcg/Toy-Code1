"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np

class Base:
    '''
    Describes the constraint base
    '''
    def __init__(self, stiffness, damping, node_ids):
        N = len(node_ids) # number of nodes involved in the constraint
        self.stiffness = stiffness
        self.damping = damping
        self.f = np.zeros((N, 2)) # forces
        self.node_ids = np.copy(node_ids) # node ids
        # Precomputed jacobians.
        # NxN matrix where each element is a 2x2 submatrix
        self.dfdx = np.zeros((N, N, 2, 2))
        self.dfdv = np.zeros((N, N, 2, 2))
