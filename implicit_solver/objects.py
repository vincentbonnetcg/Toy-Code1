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

'''
 Beam
'''
class Beam(BaseObject):
    def __init__(self, position, width, height, cellX, cellY, particleMass, stiffness, damping):
        BaseObject.__init__(self, (cellX+1)*(cellY+1), particleMass, stiffness, damping)
        
        # Set position
        # Example of vertex positions
        # 8 .. 9 .. 10 .. 11
        # 4 .. 5 .. 6  .. 7
        # 0 .. 1 .. 2  .. 3
        particleid = 0;
        cellWidth = width / cellX
        cellHeight = height / cellY
        for j in range(cellY+1):
            for i in range(cellX+1):
                self.x[particleid] = (i * cellWidth + position[0], j * cellHeight + position[1])
                particleid += 1

        # Initialize spring constraints
        cell_to_pids = lambda i, j : [i + (j*(cellX+1)) , i + (j*(cellX+1)) + 1 , i + ((j+1)*(cellX+1)), i + ((j+1)*(cellX+1)) + 1]
        for j in range(cellY):
            for i in range(cellX):
                pids = cell_to_pids(i, j)
                
                self.constraints.append(cn.SpringConstraint(stiffness, damping, [pids[1], pids[3]], self))
                if (i == 0):
                    self.constraints.append(cn.SpringConstraint(stiffness, damping, [pids[0], pids[2]], self))
                
                self.constraints.append(cn.SpringConstraint(stiffness, damping, [pids[2], pids[3]], self))
                if (j == 0): 
                    self.constraints.append(cn.SpringConstraint(stiffness, damping, [pids[0], pids[1]], self))
                    
                self.constraints.append(cn.SpringConstraint(stiffness, damping, [pids[0], pids[3]], self))
                self.constraints.append(cn.SpringConstraint(stiffness, damping, [pids[1], pids[2]], self))

        # Initialize anchor constraints
        for pid in range(0, (cellX+1)*cellY+1, cellX+1):
            self.constraints.append(cn.AnchorSpringConstraint(stiffness * 100, damping, [pid], np.copy(self.x[pid]), self))
            self.constraints.append(cn.AnchorSpringConstraint(stiffness * 100, damping, [pid+cellX], np.copy(self.x[pid+cellX]), self))
            
            
