"""
@author: Vincent Bonnet
@description : Dynamic object describes simulated objects
"""

import numpy as np
import lib.common as common
from lib.objects.components.node import Node

class Dynamic:
    '''
    Dynamic describes a simulated object
    It contains:
    Object data:
        - num_nodes
        - position: x
        - velocity: v
        - mass: m
        - inverse mass: im
        - external forces: f
    Indexing:
        - node global offset : node_global_offset
        - object index in the scene.dynamics : index
    '''
    def __init__(self, shape, node_mass):
        self.num_nodes = shape.num_vertices()

        # Allocate node data
        self.data = common.DataBlock()
        self.data.add_field_from_class(Node)
        self.data.initialize(self.num_nodes)

        # Set node data
        np.copyto(self.data.x, shape.vertex.position)
        self.data.v.fill(0.0)
        self.data.m.fill(node_mass)
        self.data.im.fill(1.0 / node_mass)
        self.data.f.fill(0.0)

        # Reference datablock attributes on the object for easy access
        self.data.set_attribute_to_object(self)

        # Initialize node connectivities
        self.edge_ids = np.copy(shape.edge.vertex_ids)
        self.face_ids = np.copy(shape.face.vertex_ids)
        # Useful indices set after adding the object into the scene
        self.node_global_offset = 0 # global offset of the node in the complete system
        self.index = 0 # object index in the scene.dynamics[.]
        # Metadata
        self.meta_data = {}

    def set_indexing(self, index, node_global_offset):
        '''
        Sets the global indices (object index and node global offset)
        Those indices are set after the object has been added to the scene
        '''
        self.index = index
        self.node_global_offset = node_global_offset

    def convert_to_shape(self):
        '''
        Create a simple shape from the dynamic datablock and
        node connectivities
        '''
        num_vertices = self.num_nodes
        num_edges = len(self.edge_ids)
        num_faces = len(self.face_ids)
        shape = common.Shape(num_vertices, num_edges, num_faces)

        for i in range(num_vertices):
            shape.vertex.position[i] = self.x[i]

        shape.edge.vertex_ids = np.copy(self.edge_ids)
        shape.face.vertex_ids = np.copy(self.face_ids)

        return shape

