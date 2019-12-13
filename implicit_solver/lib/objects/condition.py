"""
@author: Vincent Bonnet
@description : conditions create a list of constraints from a list of objects
"""

import lib.common as common
import lib.common.node_accessor as na
from lib.system import Scene

def apply_constraint_forces(constraint_blocks, details):
    for constraint_data in constraint_blocks:
        node_ids_ptr = constraint_data['node_IDs']
        force_ptr = constraint_data['f']
        block_n_elements = constraint_data['blockInfo_numElements']
        for ct_index in range(block_n_elements):
            node_ids = node_ids_ptr[ct_index]
            forces = force_ptr[ct_index]
            for i in range(len(node_ids)):
                na.node_add_f(details.node, node_ids[i], forces[i])

class Condition:
    '''
    Base of a condition
    '''
    def __init__(self, stiffness, damping, constraint_type):
        self.block_ids = []
        self.constraint_type = constraint_type
        # Parameters
        self.stiffness = stiffness
        self.damping = damping
        # Data
        self.data = common.DataBlock(constraint_type)
        # Energy / Force / Jacobian (Used by the optimiser)
        self.energy_func = None # Not used yet
        self.force_func =  constraint_type.compute_forces # derivative of the energy function
        self.jacobian_func = constraint_type.compute_jacobians # derivative of the force function
        # Metadata
        self.meta_data = {}
        self.total_constraints = 0

    def num_constraints(self) -> int:
        return self.total_constraints

    def is_static(self) -> bool:
        '''
        Returns whether or not the created constraints are dynamic or static
        Dynamic constraints are recreated every substep
        Static constraints are created at initialisation and valid for the whole simulation
        '''
        return True

    def update_constraints(self, scene : Scene, details):
        self.init_constraints(scene, details)

    def compute_forces(self, scene : Scene, details):
        self.force_func(self.data, scene, details)

    def compute_jacobians(self, scene : Scene, details):
        self.jacobian_func(self.data, scene, details)

    def apply_forces(self, details):
        apply_constraint_forces(self.data.blocks, details)

    def init_constraints(self, scene : Scene, details):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'init_constraints'")

