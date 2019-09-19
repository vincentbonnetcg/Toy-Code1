"""
@author: Vincent Bonnet
@description : conditions create a list of constraints from a list of objects
"""

import lib.common as common
import lib.common.node_accessor as na

class Condition:
    '''
    Base of a condition
    '''
    def __init__(self, stiffness, damping, constraint_type):
        # Parameters
        self.stiffness = stiffness
        self.damping = damping
        # Data
        self.data = common.DataBlock()
        self.data.add_field_from_class(constraint_type)
        # Energy / Force / Jacobian (Used by the optimiser)
        self.energy_func = None # Not used yet
        self.force_func =  constraint_type.compute_forces # derivative of the energy function
        self.jacobian_func = constraint_type.compute_jacobians # derivative of the force function
        # Metadata
        self.meta_data = {}

    def num_constraints(self) -> int:
        return self.data.num_elements

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
                na.node_add_f(scene.dynamics, node_ids[node_id], forces[node_id])

    def init_constraints(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'init_constraints'")

