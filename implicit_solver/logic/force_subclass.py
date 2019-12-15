"""
@author: Vincent Bonnet
@description : Subclasses of the Force class
"""

from lib.objects import Force
import numpy as np

def apply_gravity(node_blocks, gravity):
    for block_data in node_blocks:
        force_ptr = block_data['f']
        mass_ptr = block_data['m']
        block_n_elements = block_data['blockInfo_numElements']
        for i in range(block_n_elements):
            force_ptr[i] += gravity * mass_ptr[i]

class Gravity(Force):
    '''
    Base to describe gravity
    '''
    def __init__(self, gravity):
        self.gravity = np.array(gravity)

    def apply_forces(self, details):
        apply_gravity(details.node.blocks, self.gravity)

