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
        # Data
        self.data = core.DataBlock()
        # Energy / Force / Jacobian
        self.energy_func = None # Not used yet
        self.force_func = None # derivative of the energy function
        self.jacobian_func = None # derivative of the force function
        # Metadata
        self.meta_data = {}

    def initialize(self, constraint_type):
        '''
        Initialize the datablock field and energy functions from the constraint type
        '''
        # Initialize datablock
        num_nodes = constraint_type.num_nodes()
        self.data.add_field("stiffness", np.float)
        self.data.add_field("damping", np.float)
        self.data.add_field("node_ids", np.uint32, (num_nodes, 3)) # see Scene.node_id() to understand the 3
        self.data.add_field("f", np.float, (num_nodes, 2))
        self.data.add_field("dfdx", np.float, (num_nodes, num_nodes, 2, 2))
        self.data.add_field("dfdv", np.float, (num_nodes, num_nodes, 2, 2))
        constraint_type.add_fields(self.data)
        # Initialize functions
        self.energy_func = None
        self.force_func = constraint_type.compute_forces
        self.jacobian_func = constraint_type.compute_jacobians

    @classmethod
    def init_element(cls, element, stiffness, damping, node_ids):
        element.stiffness = stiffness
        element.damping = damping
        element.node_ids = np.copy(node_ids)

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
        self.init_constraints(scene)

    def compute_forces(self, scene):
        self.force_func(self.data, scene)

    def compute_jacobians(self, scene):
        self.jacobian_func(self.data, scene)

    def apply_forces(self, scene):
        node_ids_ptr = self.data.node_ids
        force_ptr = self.data.f

        for ct_index in range(len(self.data)):
            node_ids = node_ids_ptr[ct_index]
            forces = force_ptr[ct_index]
            for node_id in range(len(node_ids)):
                scene.node_add_f(node_ids[node_id], forces[node_id])

    def init_constraints(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'init_constraints'")

