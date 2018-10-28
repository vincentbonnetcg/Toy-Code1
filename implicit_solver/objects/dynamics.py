"""
@author: Vincent Bonnet
@description : Object descriptions for implicit solver
"""

import constraints as cn
import numpy as np
from objects.shapes import WireShape

class BaseDynamic:
    '''
    BaseDynamic describes the base class of a dynamic object
    It contains:
    Particle data:
        - num_particles
        - position: x
        - velocity: v
        - mass: m
        - inverse mass: im
        - external forces: f
    Object data / Material:
        - stiffness
        - damping
        - list of internal constraint: internal_constraints[]
    Indexing:
        - global particle offset :globalOffset
        - object index in the scene.dynamics : index
    Render Preferences
        - Render Preferences : renderPrefs
    '''
    def __init__(self, shape, particle_mass, stiffness, damping):
        self.num_particles = shape.num_vertices()
        # Initialize particle state
        self.x = np.zeros((self.num_particles, 2)) # position
        self.v = np.zeros((self.num_particles, 2)) # velocity
        self.m = np.ones(self.num_particles) * particle_mass# mass
        self.im = 1.0 / self.m # inverse mass
        self.f = np.zeros((self.num_particles, 2)) #  force
        # Useful indices set after adding the object into the scene
        self.global_offset = 0 # global particle offset
        self.index = 0 # object index in the scene.dynamics[.]
        # Material and internal constraints
        self.stiffness = stiffness
        self.damping = damping
        self.internal_constraints = []
        # Copy particle position
        for i in range(self.num_particles):
            self.x[i] = shape.vertex.position[i]
        # Render preferences used by render.py
        # See : https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html for more details
        # fmt = '[color][marker][line]'
        # format of the display State ['particle_fmt', particle_size, 'constraint_fmt', constraint_line_size ]
        self.render_prefs = ['go', 3, 'k-', 1]

    def set_indexing(self, index, global_offset):
        '''
        Sets the global indices (object index and particle offset)
        Those indices are set after the object has been added to the scene
        '''
        self.index = index
        self.global_offset = global_offset

    def create_internal_constraints(self):
        '''
        Creates the internal constraints of the dynamic object.
        It could be used to set materials / distance constraint / area constraint etc.
        '''
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'create_internal_constraints'")


class Wire(BaseDynamic):
    '''
    Wire Class describes a dynamic wire object
    '''
    def __init__(self, wire_shape, particle_mass, stiffness, bending_stiffness, damping):
        BaseDynamic.__init__(self, wire_shape, particle_mass, stiffness, damping)
        self.num_edges = wire_shape.num_edges()
        self.bending_stiffness = bending_stiffness

    def create_internal_constraints(self):
        for i in range(self.num_edges):
            self.internal_constraints.append(cn.Spring(self.stiffness, self.damping, [self, self], [i, i+1]))

        if self.num_edges > 1 and self.bending_stiffness > 0.0:
            for i in range(self.num_edges-1):
                self.internal_constraints.append(cn.Bending(self.bending_stiffness, self.damping, [self, self, self], [i, i+1, i+2]))

class Beam(BaseDynamic):
    '''
    Beam Class describes a dynamic beam object
    '''
    def __init__(self,beam_shape , particle_mass, stiffness, damping):
        BaseDynamic.__init__(self, beam_shape, particle_mass, stiffness, damping)

        self.cell_x = beam_shape.cell_x
        self.cell_y = beam_shape.cell_y

    def create_internal_constraints(self):
        cell_to_pids = lambda i, j: [i + (j*(self.cell_x+1)), i + (j*(self.cell_x+1)) + 1, i + ((j+1)*(self.cell_x+1)), i + ((j+1)*(self.cell_x+1)) + 1]
        # Compute Spring Constraint
        for j in range(self.cell_y):
            for i in range(self.cell_x):
                pids = cell_to_pids(i, j)

                self.internal_constraints.append(cn.Spring(self.stiffness, self.damping, [self, self], [pids[1], pids[3]]))
                if i == 0:
                    self.internal_constraints.append(cn.Spring(self.stiffness, self.damping, [self, self], [pids[0], pids[2]]))

                self.internal_constraints.append(cn.Spring(self.stiffness, self.damping, [self, self], [pids[2], pids[3]]))
                if j == 0:
                    self.internal_constraints.append(cn.Spring(self.stiffness, self.damping, [self, self], [pids[0], pids[1]]))

                #self.internal_constraints.append(cn.Spring(self.stiffness, self.damping, [self, self], [pids[0], pids[3]]))
                #self.internal_constraints.append(cn.Spring(self.stiffness, self.damping, [self, self], [pids[1], pids[2]]))

        # Compute Area Constraint
        for j in range(self.cell_y):
            for i in range(self.cell_x):
                pids = cell_to_pids(i, j)

                self.internal_constraints.append(cn.Area(self.stiffness, self.damping, [self, self, self], [pids[0], pids[1], pids[2]]))
                self.internal_constraints.append(cn.Area(self.stiffness, self.damping, [self, self, self], [pids[1], pids[2], pids[3]]))
