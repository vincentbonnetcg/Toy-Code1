"""
@author: Vincent Bonnet
@description : Shape description
"""

import numpy as np

class VertexComponent:
    '''
    Vertex Component
    '''
    def __init__(self, num_vertices):
        self.position = np.zeros((num_vertices, 2), dtype=float)

    def __len__(self):
        return len(self.position)

class EdgeComponent:
    '''
    Edge Component
    '''
    def __init__(self, num_edges):
        self.vertex_ids = np.zeros((num_edges, 2), dtype=int)

    def __len__(self):
        return len(self.vertex_ids)

    def vertex_edges_dict(self):
        return component_to_vertex_ids(self.vertex_ids)

class FaceComponent:
    '''
    Face Component
    '''
    def __init__(self, num_faces):
        self.vertex_ids = np.zeros((num_faces, 3), dtype=int)

    def __len__(self):
        return len(self.vertex_ids)

def component_to_vertex_ids(vertex_ids):
    result = {}
    for component_id, vertex_ids in enumerate(vertex_ids):
        for vertex_id in vertex_ids:
            result.setdefault(vertex_id, []).append(component_id)

    return result

def vertex_ids_neighbours(vertex_ids):
    result = {}
    for component_id, vertex_ids in enumerate(vertex_ids):
        for vertex_id0 in vertex_ids:
            for vertex_id1 in vertex_ids:
                if vertex_id0 != vertex_id1:
                    result.setdefault(vertex_id0, []).append(vertex_id1)
    return result

class Shape:
    '''
    Shape Description
    '''
    def __init__(self, num_vertices, num_edges=0, num_faces=0):
        self.vertex = VertexComponent(num_vertices)
        self.edge = EdgeComponent(num_edges)
        self.face = FaceComponent(num_faces)

    def extract_transform_from_shape(self):
        '''
        Returns the 'optimal' position and modify the shape vertices from world space to local space
        Optimal rotation is not computed
        '''
        optimal_pos = np.average(self.vertex.position, axis=0)
        np.subtract(self.vertex.position, optimal_pos, out=self.vertex.position)
        optimal_rot = 0
        return optimal_pos, optimal_rot

    def num_vertices(self):
        return len(self.vertex)

    def num_edges(self):
        return len(self.edge)

    def num_faces(self):
        return len(self.face)
