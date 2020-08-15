"""
@author: Vincent Bonnet
@description : Constraint data structure
"""

import numpy as np
import lib.common.jit.data_accessor as db

class Constraint:
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

class AnchorSpring(Constraint):
    def __init__(self):
        Constraint.__init__(self, num_nodes = 1)
        self.rest_length = np.float64(0.0)
        self.kinematic_component_IDs = db.empty_data_ids(2) # Point ids
        self.kinematic_component_param = np.float64(0.0)
        self.kinematic_component_pos = np.zeros(2, dtype = np.float64)

class Spring(Constraint):
    def __init__(self):
        Constraint.__init__(self, num_nodes = 2)
        self.rest_length = np.float64(0.0)

class Bending(Constraint):
    def __init__(self):
        # Maintain angle between (x0,x1) and (x1,x2)
        Constraint.__init__(self, num_nodes = 3)
        self.rest_angle = np.float64(0.0)

class Area(Constraint):
    def __init__(self):
        Constraint.__init__(self, num_nodes = 3)
        self.rest_area = np.float64(0.0)
