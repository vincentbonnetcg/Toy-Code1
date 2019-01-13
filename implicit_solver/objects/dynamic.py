"""
@author: Vincent Bonnet
@description : Dynamic object describes simulated objects
"""

import numpy as np
from core.data_block import DataBlock

class Dynamic:
    '''
    Dynamic describes a simulated object
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

        # Create particle data
        self.data = DataBlock()
        self.data.add_field("x", np.float, 2)
        self.data.add_field("v", np.float, 2)
        self.data.add_field("f", np.float, 2)
        self.data.add_field("m", np.float)
        self.data.add_field("im", np.float)
        self.data.initialize(self.num_particles)
        np.copyto(self.data.x, shape.vertex.position)
        self.data.m.fill(particle_mass)
        self.data.im.fill(1.0 / particle_mass)

        # Reference on particle state for easy access
        self.x = self.data.x # position
        self.v = self.data.v # velocity
        self.m = self.data.m # mass
        self.im = self.data.im # inverse mass
        self.f = self.data.f # forces

        # Initialize particle connectivities
        self.edge_ids = np.copy(shape.edge.vertex_ids)
        self.face_ids = np.copy(shape.face.vertex_ids)
        # Useful indices set after adding the object into the scene
        self.global_offset = 0 # global particle offset
        self.index = 0 # object index in the scene.dynamics[.]
        # Metadata
        self.meta_data = {}

    def set_indexing(self, index, global_offset):
        '''
        Sets the global indices (object index and particle offset)
        Those indices are set after the object has been added to the scene
        '''
        self.index = index
        self.global_offset = global_offset
