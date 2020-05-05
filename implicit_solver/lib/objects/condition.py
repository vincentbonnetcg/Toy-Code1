"""
@author: Vincent Bonnet
@description : conditions create a list of constraints from a list of objects
"""

import lib.common.jit.block_utils as block_utils

class Condition:
    '''
    Base of a condition
    '''
    def __init__(self, stiffness, damping, constraint_type):
        self.block_handles = block_utils.empty_block_handles()
        self.constraint_type = constraint_type
        # Parameters
        self.stiffness = stiffness
        self.damping = damping
        # Metadata
        self.meta_data = {}
        self.total_constraints = 0

    def num_constraints(self) -> int:
        return self.total_constraints

    def num_blocks(self) -> int:
        return len(self.block_handles)

    def is_static(self) -> bool:
        '''
        Returns whether or not the created constraints are dynamic or static
        Dynamic constraints are recreated every substep
        Static constraints are created at initialisation and valid for the whole simulation
        '''
        return True

    def init_constraints(self, scene, details):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'init_constraints'")

    def update_constraints(self, scene, details):
        pass

    def __call_func(self, func, details, use_point = False):
        if func and len(self.block_handles)>0:
            data = details.block_from_datatype(self.constraint_type)
            if use_point:
                func(data, details.node, details.point, self.block_handles)
            else:
                func(data, details.node, self.block_handles)

    def pre_compute(self, details):
        self.__call_func(self.constraint_type.pre_compute(), details, use_point=True)

    def compute_rest(self, details):
        self.__call_func(self.constraint_type.compute_rest(), details)

    def compute_gradients(self, details):
        self.__call_func(self.constraint_type.compute_gradients(), details)

    def compute_hessians(self, details):
        self.__call_func(self.constraint_type.compute_hessians(), details)

