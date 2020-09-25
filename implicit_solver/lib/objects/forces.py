"""
@author: Vincent Bonnet
@description : External forces
"""


import numpy as np
import numba # required by core.code_gen

from lib.objects.jit.data import Node
import core.code_gen as generate

class Force:
    '''
    Base to describe a global external force
    '''
    def __init__(self):
        pass

    def apply_forces(self, nodes):
        pass

@generate.vectorize
def apply_gravity(node : Node, gravity):
    node.f += gravity * node.m

class Gravity(Force):
    '''
    Base to describe gravity
    '''
    def __init__(self, gravity):
        self.gravity = np.array(gravity)

    def apply_forces(self, nodes):
        apply_gravity(nodes, self.gravity)
