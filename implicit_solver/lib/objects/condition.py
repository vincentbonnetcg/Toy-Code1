"""
@author: Vincent Bonnet
@description : conditions create a list of constraints from a list of objects
"""

import lib.common as common
import lib.common.node_accessor as na
from lib.system.scene import Scene

def apply_constraint_forces(constraint_blocks, dynamics):
    for constraint_data in constraint_blocks:
        node_ids_ptr = constraint_data['node_ids']
        force_ptr = constraint_data['f']
        block_n_elements = constraint_data['blockInfo_numElements']
        for ct_index in range(block_n_elements):
            node_ids = node_ids_ptr[ct_index]
            forces = force_ptr[ct_index]
            for node_id in range(len(node_ids)):
                na.node_add_f(dynamics, node_ids[node_id], forces[node_id])

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

    def update_constraints(self, scene : Scene):
        self.init_constraints(scene)

    def compute_forces(self, scene : Scene):
        self.force_func(self.data, scene)

    def compute_jacobians(self, scene : Scene):
        self.jacobian_func(self.data, scene)

    def apply_forces(self, dynamics):
        self.data.update_blocks_from_data()
        apply_constraint_forces(self.data.blocks, dynamics)
        self.data.update_data_from_blocks()

    def init_constraints(self, scene : Scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'init_constraints'")

