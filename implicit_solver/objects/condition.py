"""
@author: Vincent Bonnet
@description : conditions create a list of constraints from a list of objects
"""

import numpy as np
import core

class Condition:
    '''
    Base of a condition
    '''
    def __init__(self, dynamics, kinematics, stiffness, damping):
        '''
        dynamics and kinematics are the objects involved in the constraint
        stiffness and damping are the constraint parameters
        '''
        # Parameters
        self.stiffness = stiffness
        self.damping = damping
        self.dynamic_indices = [dynamic.index for dynamic in dynamics]
        self.kinematic_indices = [kinematic.index for kinematic in kinematics]
        self.constraints = []
        # Data
        self.data = core.DataBlock()
        # Metadata
        self.meta_data = {}

    def initialize_datablock(self, constraint_type):
        '''
        Initialize the datablock field from a constraint type
        '''
        num_nodes = constraint_type.num_nodes()
        self.data.add_field("stiffness", np.float)
        self.data.add_field("damping", np.float)
        self.data.add_field("node_ids", np.uint32, (num_nodes, 3)) # see Scene.node_id() to understand the 3
        self.data.add_field("f", np.float, (num_nodes, 2))
        self.data.add_field("dfdx", np.float, (num_nodes, num_nodes, 2, 2))
        self.data.add_field("dfdv", np.float, (num_nodes, num_nodes, 2, 2))
        constraint_type.add_fields(self.data)

    def num_constraints(self) -> int:
        return len(self.data)

    def is_static(self) -> bool:
        '''
        Returns whether or not the created constraints are dynamic or static
        Dynamic constraints are recreated every substep
        Static constraints are created at initialisation and valid for the whole simulation
        '''
        return True

    def update_constraints(self, scene):
        self.constraints.clear()
        self.add_constraints(scene)

    def add_constraints(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'add_constraints'")

