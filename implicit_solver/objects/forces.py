"""
@author: Vincent Bonnet
@description : External forces
"""

import numpy as np

class Force:
    '''
    Base to describe a global external force
    '''
    def __init__(self):
        pass

    def apply_forces(self, scene):
        pass

class Gravity:
    '''
    Base to describe gravity
    '''
    def __init__(self, gravity):
        self.gravity = gravity

    def apply_forces(self, scene):
        for dynamic in scene.dynamics:
            dynamic.f.fill(0.0)
            for i in range(dynamic.num_particles):
                dynamic.f[i] += np.multiply(self.gravity, dynamic.m[i])

