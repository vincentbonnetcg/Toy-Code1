"""
@author: Vincent Bonnet
@description : Subclasses of the Force class
"""

from lib.objects import Force
import numpy as np

def apply_gravity(dynamic_blocks, gravity):
    for dynamic_data in dynamic_blocks:
        force_ptr = dynamic_data['f']
        mass_ptr = dynamic_data['m']
        block_n_elements = dynamic_data['blockInfo_numElements']
        for i in range(block_n_elements):
            force_ptr[i] += gravity * mass_ptr[i]

class Gravity(Force):
    '''
    Base to describe gravity
    '''
    def __init__(self, gravity):
        self.gravity = np.array(gravity)

    def apply_forces(self, dynamics):
        for dynamic in dynamics:
            dynamic.data.update_blocks_from_data()
            apply_gravity(dynamic.data.blocks, self.gravity)
            dynamic.data.update_data_from_blocks()

