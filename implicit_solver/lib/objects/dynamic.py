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
    '''
    def __init__(self, shape, node_mass):
        num_nodes = shape.num_vertices()

        # Allocate node data
        self.data = common.DataBlock()
        self.data.add_field_from_class(Node)
        self.data.initialize(num_nodes)

        # Set node data
        self.data.copyto('x', shape.vertex)
        self.data.fill('v', 0.0)
        self.data.fill('m', node_mass)
        self.data.fill('im', 1.0 / node_mass)
        self.data.fill('f', 0.0)
        self.data.fill('ID', 0)

        # Initialize node connectivities
        self.edge_ids = np.copy(shape.edge)
        self.face_ids = np.copy(shape.face)
        # Object index in the scene.dynamics[.]
        self.index = 0
        # Metadata
        self.meta_data = {}

    def num_nodes(self) -> int:
        return self.data.num_elements

    def set_indexing(self, object_id, node_global_offset):
        '''
        Sets the global indices (object index and node global offset)
        Those indices are set after the object has been added to the scene
        '''
        self.data.set_indexing(object_id, node_global_offset)
        self.index = object_id

    def convert_to_shape(self):
        '''
        Create a simple shape from the dynamic datablock and
        node connectivities
        '''
        num_vertices = self.num_nodes()
        num_edges = len(self.edge_ids)
        num_faces = len(self.face_ids)
        shape = common.Shape(num_vertices, num_edges, num_faces)
        shape.vertex = self.data.flatten('x')
        shape.edge = np.copy(self.edge_ids)
        shape.face = np.copy(self.face_ids)

        return shape

