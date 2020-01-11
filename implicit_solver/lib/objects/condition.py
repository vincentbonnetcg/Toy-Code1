"""
@author: Vincent Bonnet
@description : conditions create a list of constraints from a list of objects
"""

from lib.system import Scene
import numpy as np
import lib.common as common

class Condition:
    '''
    Base of a condition
    '''
    def __init__(self, stiffness, damping, constraint_type):
        self.block_handles = common.DataBlock.create_block_handle(None)
        self.constraint_type = constraint_type
        # Parameters
        self.stiffness = stiffness
        self.damping = damping
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
        if len(self.block_handles)>0:
            func = self.constraint_type.pre_compute()
            if func:
                data = details.block_from_datatype(self.constraint_type)
                func(data, scene, details.node, self.block_handles)

    def compute_rest(self, details):
        if len(self.block_handles)>0:
            func = self.constraint_type.compute_rest()
            if func:
                data = details.block_from_datatype(self.constraint_type)
                func(data, details.node, self.block_handles)

    def compute_gradients(self, details):
        if len(self.block_handles)>0:
            func = self.constraint_type.compute_gradients()
            if func:
                data = details.block_from_datatype(self.constraint_type)
                func(data, details.node, self.block_handles)

    def compute_hessians(self, details):
        if len(self.block_handles)>0:
            func = self.constraint_type.compute_hessians()
            if func:
                data = details.block_from_datatype(self.constraint_type)
                func(data, details.node, self.block_handles)
