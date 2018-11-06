"""
@author: Vincent Bonnet
@description : Object descriptions for implicit solver
"""

import numpy as np

class Dynamic:
    '''
    Dynamic describes the base class of a dynamic object
    It contains:
    Object data:
        - num_particles
        - position: x
        - velocity: v
        - mass: m
        - inverse mass: im
        - external forces: f
    Indexing:
        - global particle offset :globalOffset
        - object index in the scene.dynamics : index
    Render Preferences
        - Render Preferences : renderPrefs
    '''
    def __init__(self, shape, particle_mass):
        self.num_particles = shape.num_vertices()
        # Initialize particle state
        self.x = np.zeros((self.num_particles, 2)) # position
        self.v = np.zeros((self.num_particles, 2)) # velocity
        self.m = np.ones(self.num_particles) * particle_mass# mass
        self.im = 1.0 / self.m # inverse mass
        self.f = np.zeros((self.num_particles, 2)) #  force
        # Initialize particle connectivities
        self.edge_ids = np.copy(shape.edge.vertex_ids)
        self.face_ids = np.copy(shape.face.vertex_ids)
        # Useful indices set after adding the object into the scene
        self.global_offset = 0 # global particle offset
        self.index = 0 # object index in the scene.dynamics[.]
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
