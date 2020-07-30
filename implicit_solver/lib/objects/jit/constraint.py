"""
@author: Vincent Bonnet
@description : Constraint base for the implicit solver
"""

import numpy as np

import lib.common.jit.data_accessor as db

class Constraint:
    '''
    Describes the constraint base
    '''
    def __init__(self, num_nodes : int):

        # Constraint Property
        self.stiffness = np.float64(0.0)
        self.damping = np.float64(0.0)

        # Node ids involved in the constraint
        self.node_IDs = db.empty_data_ids(num_nodes)

        # system indices of the nodes
        self.systemIndices = np.zeros(num_nodes, dtype = np.int32)

        # Precomputed cost function
        self.c = np.zeros(num_nodes, dtype = np.float64) # constraint/cost function
        self.g = np.zeros((num_nodes, 2), dtype = np.float64) # gradients
        self.H = np.zeros((num_nodes, num_nodes, 2, 2), dtype = np.float64) # Hessians

        # Precomputed forces/jacobians.
        self.f = np.zeros((num_nodes, 2), dtype = np.float64)
        self.dfdx = np.zeros((num_nodes, num_nodes, 2, 2), dtype = np.float64)
        self.dfdv = np.zeros((num_nodes, num_nodes, 2, 2), dtype = np.float64)
