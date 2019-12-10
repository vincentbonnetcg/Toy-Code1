"""
@author: Vincent Bonnet
@description : Dynamic object describes simulated objects
"""

import numpy as np
import lib.common as common
from lib.objects.components.node import Node
import lib.common.node_accessor as na

class Dynamic:
    '''
    Dynamic describes a simulated object
    '''
    def __init__(self, details, shape, node_mass):
        # Allocate node data
        self.total_nodes = shape.num_vertices()

        self.data = common.DataBlock(Node) # Old
        self.blocks_ids = self.data.append(self.total_nodes)
        self.node_ids = None

        # TODO - replace local datablock with details
        #self.node_ids = details.node.append(self.total_nodes)

        # Set node data
        self.data.copyto('x', shape.vertex, self.blocks_ids)
        self.data.fill('v', 0.0, self.blocks_ids)
        self.data.fill('m', node_mass, self.blocks_ids)
        self.data.fill('im', 1.0 / node_mass, self.blocks_ids)
        self.data.fill('f', 0.0, self.blocks_ids)

        # Initialize node connectivities
        self.edge_ids = np.copy(shape.edge)
        self.face_ids = np.copy(shape.face)
        # Object index in the scene.dynamics[.]
        self.index = 0
        # Metadata
        self.meta_data = {}

    def num_nodes(self) -> int:
        return self.total_nodes

    def set_indexing(self, object_id, node_global_offset):
        '''
        Sets the global indices (object index and node global offset)
        Those indices are set after the object has been added to the scene
        '''
        self.index = object_id
        self.node_ids  = self.data.flatten('ID', self.blocks_ids)
        for vertex_index, node_id in enumerate(self.node_ids):
            na.set_object_id(node_id, self.index, node_global_offset+vertex_index)
        self.data.copyto('ID', self.node_ids, self.blocks_ids)

    def get_node_id(self, vertex_index):
        return self.node_ids[vertex_index]

    def convert_to_shape(self):
        '''
        Create a simple shape from the dynamic datablock and
        node connectivities
        '''
        num_vertices = self.num_nodes()
        num_edges = len(self.edge_ids)
        num_faces = len(self.face_ids)
        shape = common.Shape(num_vertices, num_edges, num_faces)
        shape.vertex = self.data.flatten('x', self.blocks_ids)
        shape.edge = np.copy(self.edge_ids)
        shape.face = np.copy(self.face_ids)

        return shape

