"""
@author: Vincent Bonnet
@description : Dynamic object representation
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
    '''
    def __init__(self, shape, particle_mass):
        self.num_particles = shape.num_vertices()
        # Initialize particle state
        self.x = np.copy(shape.vertex.position) # position
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

    def set_indexing(self, index, global_offset):
        '''
        Sets the global indices (object index and particle offset)
        Those indices are set after the object has been added to the scene
        '''
        self.index = index
        self.global_offset = global_offset
