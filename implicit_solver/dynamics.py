"""
@author: Vincent Bonnet
@description : Object descriptions for implicit solver
"""

import constraints as cn
import numpy as np

'''
 Base Dynamic
'''
class BaseDynamic:
    def __init__(self, numParticles, particleMass, stiffness, damping):
        self.numParticles = numParticles
        # Initialize particle state
        self.x = np.zeros((self.numParticles, 2)) # position
        self.v = np.zeros((self.numParticles, 2)) # velocity
        self.m = np.ones(self.numParticles) * particleMass# mass
        self.im = 1.0 / self.m # inverse mass
        self.f = np.zeros((self.numParticles, 2)) #  force
        # Useful indices set after adding the object into the scene
        self.globalOffset = 0
        self.index = 0 # index in the scene.dynamics[]
        # Material property
        self.stiffness = stiffness
        self.damping = damping
        
        # Initialize constraints
        self.constraints = []
        
        # Render preferences used by render.py
        # See : https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html for more details
        # fmt = '[color][marker][line]'
        # format of the display State ['particle_fmt', particle_size, 'constraint_fmt', constraint_line_size ]
        self.renderPrefs = ['go', 3, 'k-', 1]
    
    def createInternalConstraints(self):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'createInternalConstraints'")

'''
 Wire
'''
class Wire(BaseDynamic):
    def __init__(self, startPoint, endPoint, numEdges, particleMass, stiffness, damping):
        BaseDynamic.__init__(self, numEdges+1, particleMass, stiffness, damping)
        self.numEdges = numEdges
        
        axisx = np.linspace(startPoint[0], endPoint[0], num=self.numParticles, endpoint=True)
        axisy = np.linspace(startPoint[1], endPoint[1], num=self.numParticles, endpoint=True)
        for i in range(self.numParticles):
            self.x[i] = (axisx[i], axisy[1])

    def createInternalConstraints(self):
        for i in range(self.numEdges):
            self.constraints.append(cn.SpringConstraint(self.stiffness, self.damping, [self, self], [i, i+1]))

'''
 Beam
'''
class Beam(BaseDynamic):
    def __init__(self, position, width, height, cellX, cellY, particleMass, stiffness, damping):
        BaseDynamic.__init__(self, (cellX+1)*(cellY+1), particleMass, stiffness, damping)
        
        # Set position
        # Example of vertex positions
        # 8 .. 9 .. 10 .. 11
        # 4 .. 5 .. 6  .. 7
        # 0 .. 1 .. 2  .. 3
        self.cellX = cellX
        self.cellY = cellY
        particleId = 0;
        cellWidth = width / cellX
        cellHeight = height / cellY
        for j in range(cellY+1):
            for i in range(cellX+1):
                self.x[particleId] = (i * cellWidth + position[0], j * cellHeight + position[1])
                particleId += 1

    def createInternalConstraints(self):
        cell_to_pids = lambda i, j : [i + (j*(self.cellX+1)) , i + (j*(self.cellX+1)) + 1 , i + ((j+1)*(self.cellX+1)), i + ((j+1)*(self.cellX+1)) + 1]
        for j in range(self.cellY):
            for i in range(self.cellX):
                pids = cell_to_pids(i, j)
                
                self.constraints.append(cn.SpringConstraint(self.stiffness, self.damping, [self, self], [pids[1], pids[3]]))
                if (i == 0):
                    self.constraints.append(cn.SpringConstraint(self.stiffness, self.damping, [self, self], [pids[0], pids[2]]))
                
                self.constraints.append(cn.SpringConstraint(self.stiffness, self.damping, [self, self], [pids[2], pids[3]]))
                if (j == 0): 
                    self.constraints.append(cn.SpringConstraint(self.stiffness, self.damping, [self, self], [pids[0], pids[1]]))
                    
                self.constraints.append(cn.SpringConstraint(self.stiffness, self.damping, [self, self], [pids[0], pids[3]]))
                self.constraints.append(cn.SpringConstraint(self.stiffness, self.damping, [self, self], [pids[1], pids[2]]))
