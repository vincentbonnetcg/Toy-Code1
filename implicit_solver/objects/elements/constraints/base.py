"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np

class Base:
    '''
    Describes the constraint base
    '''
    def __init__(self, num_nodes : int):

        # Constraint Property
        self.stiffness = np.float32(0.0)
        self.damping = np.float32(0.0)

        # Node ids involved in the constraint
        # Should match result size of scene.node_id() (three unsigned)
        self.node_ids = np.zeros((num_nodes, 3), dtype=np.uint32)

        # Precomputed forces/jacobians.
        self.f = np.zeros((num_nodes, 2), dtype = np.float32)
        self.dfdx = np.zeros((num_nodes, num_nodes, 2, 2), dtype = np.float32)
        self.dfdv = np.zeros((num_nodes, num_nodes, 2, 2), dtype = np.float32)
