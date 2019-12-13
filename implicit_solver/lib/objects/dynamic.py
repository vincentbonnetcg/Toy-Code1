"""
@author: Vincent Bonnet
@description : Dynamic object describes simulated objects
"""

import numpy as np
import lib.common as common

class Dynamic:
    '''
    Dynamic describes a simulated object
    '''
    def __init__(self, details, shape, node_mass):
        # Allocate node data
        self.total_nodes = shape.num_vertices()
        self.blocks_ids = details.node.append(self.total_nodes)
        self.node_ids = details.node.flatten('ID', self.blocks_ids)

        # Set node data
        details.node.copyto('x', shape.vertex, self.blocks_ids)
        details.node.fill('v', 0.0, self.blocks_ids)
        details.node.fill('m', node_mass, self.blocks_ids)
        details.node.fill('im', 1.0 / node_mass, self.blocks_ids)
        details.node.fill('f', 0.0, self.blocks_ids)

        # Initialize node connectivities
        self.edge_ids = np.copy(shape.edge)
        self.face_ids = np.copy(shape.face)
        # Object index in the scene.dynamics[.]
        self.index = 0
        # Metadata
        self.meta_data = {}

    def num_nodes(self) -> int:
        return self.total_nodes

    def set_indexing(self, index):
        self.index = index

    def get_node_id(self, vertex_index):
        return self.node_ids[vertex_index]

    def convert_to_shape(self, details):
        '''
        Create a simple shape from the dynamic datablock and
        node connectivities
        '''
        num_vertices = self.num_nodes()
        num_edges = len(self.edge_ids)
        num_faces = len(self.face_ids)
        shape = common.Shape(num_vertices, num_edges, num_faces)
        shape.vertex = details.node.flatten('x', self.blocks_ids)
        shape.edge = np.copy(self.edge_ids)
        shape.face = np.copy(self.face_ids)

        return shape

