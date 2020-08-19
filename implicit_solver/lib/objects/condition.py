"""
@author: Vincent Bonnet
@description : conditions create a list of constraints from a list of objects
"""

import lib.common.jit.block_utils as block_utils

class Condition:
    class FunctionBundle:
        def __init__(self):
            self.pre_compute = None
            self.compute_rest = None
            self.compute_function = None
            self.compute_gradients = None
            self.compute_hessians = None
            self.compute_forces = None
            self.compute_force_jacobians = None

    def __init__(self, stiffness, damping, constraint_type):
        self.block_handles = block_utils.empty_block_handles()
        self.typename = constraint_type.name()
        # Parameters
        self.stiffness = stiffness
        self.damping = damping
        # Metadata
        self.meta_data = {}
        self.total_constraints = 0
        # Function bundle
        self.func = Condition.FunctionBundle()

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

    def init_constraints(self, details):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'init_constraints'")

    def update_constraints(self, details):
        pass

    def __call_func(self, func, details):
        if func and len(self.block_handles)>0:
            blocks = getattr(details, self.typename)
            func.function(blocks, details, self.block_handles)

    # initialization functions
    def pre_compute(self, details):
        self.__call_func(self.func.pre_compute, details)

    def compute_rest(self, details):
        self.__call_func(self.func.compute_rest, details)

    # constraint functions (function, gradients, hessians)
    def compute_function(self, details):
        self.__call_func(self.func.compute_function, details)

    def compute_gradients(self, details):
        self.__call_func(self.func.compute_gradients, details)

    def compute_hessians(self, details):
        self.__call_func(self.func.compute_hessians, details)

    # force functions (forces and their jacobians)
    def compute_forces(self, details):
        self.__call_func(self.func.compute_forces, details)

    def compute_force_jacobians(self, details):
        self.__call_func(self.func.compute_force_jacobians, details)

