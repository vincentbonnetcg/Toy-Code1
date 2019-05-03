"""
@author: Vincent Bonnet
@description : Subclasses of the Force class
"""

from objects import Force
import numpy as np

class Gravity(Force):
    '''
    Base to describe gravity
    '''
    def __init__(self, gravity):
        self.gravity = gravity

    def apply_forces(self, scene):
        for dynamic in scene.dynamics:
            dynamic.f.fill(0.0)
            for i in range(dynamic.num_nodes):
                dynamic.f[i] += np.multiply(self.gravity, dynamic.m[i])

