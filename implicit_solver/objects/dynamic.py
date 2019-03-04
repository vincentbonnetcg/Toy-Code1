"""
@author: Vincent Bonnet
@description : Dynamic object describes simulated objects
"""

import numpy as np
import core

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
        self.data = core.DataBlock()
        self.data.add_field("x", np.float, 2)
        self.data.add_field("v", np.float, 2)
        self.data.add_field("f", np.float, 2)
        self.data.add_field("m", np.float)
        self.data.add_field("im", np.float)
        self.data.initialize(self.num_particles)
        np.copyto(self.data.x, shape.vertex.position)
        self.data.m.fill(particle_mass)
        self.data.im.fill(1.0 / particle_mass)

        # Reference particle attribute for easy access
        self.data.set_attribute_to_object(self)

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

    def convert_to_shape(self):
        '''
        Create a simple shape from the dynamic datablock and
        particle connectivities
        '''
        num_vertices = self.num_particles
        num_edges = len(self.edge_ids)
        num_faces = len(self.face_ids)
        shape = core.Shape(num_vertices, num_edges, num_faces)

        for i in range(num_vertices):
            shape.vertex.position[i] = self.x[i]

        shape.edge.vertex_ids = np.copy(self.edge_ids)
        shape.face.vertex_ids = np.copy(self.face_ids)

        return shape

