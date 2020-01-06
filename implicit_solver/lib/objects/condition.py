"""
@author: Vincent Bonnet
@description : conditions create a list of constraints from a list of objects
"""

from lib.system import Scene

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
        # Energy / Force / Jacobian (Used by the optimiser)
        self.energy_func = None # Not used yet
        self.force_func =  constraint_type.compute_forces # derivative of the energy function
        self.jacobian_func = constraint_type.compute_jacobians # derivative of the force function
        self.pre_compute_func = constraint_type.pre_compute # pre compute whatever is needed. can be empty
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

    def init_constraints(self, scene : Scene, details):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'init_constraints'")

    def update_constraints(self, scene : Scene, details):
        pass

    def pre_compute(self, scene : Scene, details):
        blocks_iterator = details.block_from_datatype(self.constraint_type).get_blocks(self.block_ids)
        self.pre_compute_func(blocks_iterator, scene, details)

    def compute_forces(self, details):
        blocks_iterator = details.block_from_datatype(self.constraint_type).get_blocks(self.block_ids)
        self.force_func(blocks_iterator, details)

    def compute_jacobians(self, details):
        blocks_iterator = details.block_from_datatype(self.constraint_type).get_blocks(self.block_ids)
        self.jacobian_func(blocks_iterator, details)

