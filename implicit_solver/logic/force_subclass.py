"""
@author: Vincent Bonnet
@description : Subclasses of the Force class
"""

from lib.objects import Force
import numpy as np

class Gravity(Force):
    '''
    Base to describe gravity
    '''
    def __init__(self, gravity):
        self.gravity = gravity

    def apply_forces(self, dynamics):
        for dynamic in dynamics:
            dynamic.data.f.fill(0.0)
            for i in range(dynamic.num_nodes()):
                dynamic.data.f[i] += np.multiply(self.gravity, dynamic.data.m[i])
