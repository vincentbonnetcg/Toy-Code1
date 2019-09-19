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
        self.num_nodes = shape.num_vertices()

        # Allocate node data
        self.data = common.DataBlock()
        self.data.add_field_from_class(Node)
        self.data.initialize(self.num_nodes)

        # Set node data
        self.data.copyto('x', shape.vertex.position)
        self.data.fill('v', 0.0)
        self.data.fill('m', node_mass)
        self.data.fill('im', 1.0 / node_mass)
        self.data.fill('f', 0.0)
        self.data.fill('node_id', 0.0)
        self.data.update_data_from_blocks()

        # Initialize node connectivities
        self.edge_ids = np.copy(shape.edge.vertex_ids)
        self.face_ids = np.copy(shape.face.vertex_ids)
        # Object index in the scene.dynamics[.]
        self.index = 0
        # Metadata
        self.meta_data = {}

    def set_indexing(self, object_id, node_global_offset):
        '''
        Sets the global indices (object index and node global offset)
        Those indices are set after the object has been added to the scene
        '''
        for local_node_id in range(len(self.data)):
            global_node_id = node_global_offset + local_node_id
            self.data.node_id[local_node_id] = [object_id, local_node_id, global_node_id]
        self.index = object_id


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

