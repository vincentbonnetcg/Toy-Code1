"""
@author: Vincent Bonnet
@description : Subclasses of the Force class
"""

import numpy as np
import numba # required by lib.common.code_gen

from lib.objects import Force
import lib.objects.jit as cpn
import lib.common.code_gen as generate

@generate.as_vectorized
def apply_gravity(node : cpn.Node, gravity):
    node.f += gravity * node.m

class Gravity(Force):
    '''
    Base to describe gravity
    '''
    def __init__(self, gravity):
        self.gravity = np.array(gravity)

    def apply_forces(self, nodes):
        apply_gravity(nodes, self.gravity)
