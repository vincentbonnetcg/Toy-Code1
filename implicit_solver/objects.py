"""
@author: Vincent Bonnet
@description : Object descriptions for implicit solver
"""

import constraints as cn
import numpy as np

'''
 Base Object
'''
class BaseObject:
    def __init__(self, numParticles, particleMass, stiffness, damping):
        self.numParticles = numParticles
        # Initialize particle state
        self.x = np.zeros((self.numParticles, 2)) # position
        self.v = np.zeros((self.numParticles, 2)) # velocity
        self.m = np.ones(self.numParticles) * particleMass# mass
        self.im = 1.0 / self.m # inverse mass
        self.f = np.zeros((self.numParticles, 2)) #  force
        
        # Initialize constraints
        self.constraints = []

'''
 Wire
'''
class Wire(BaseObject):
    def __init__(self, root, length, numEdges, particleMass, stiffness, damping):
        BaseObject.__init__(self, numEdges+1, particleMass, stiffness, damping)
        self.numEdges = numEdges
        
        # Set position : start the rod in a horizontal position
        axisx = np.linspace(root[0], root[0]+length, num=self.numParticles, endpoint=True)
        for i in range(self.numParticles):
            self.x[i] = (axisx[i], root[1])

        # Initialize constraints
        self.constraints.append(cn.AnchorSpringConstraint(stiffness, damping, [0], root, self))
        for i in range(self.numEdges):
            self.constraints.append(cn.SpringConstraint(stiffness, damping, [i, i+1], self))

