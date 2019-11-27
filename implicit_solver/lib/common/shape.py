"""
@author: Vincent Bonnet
@description : Shape description
"""

import numpy as np

def vertex_ids_neighbours(vertex_ids):
    result = {}
    for component_id, vtx_ids in enumerate(vertex_ids):
        for vtx_id0 in vtx_ids:
            for vtx_id1 in vtx_ids:
                if vtx_id0 != vtx_id1:
                    result.setdefault(vtx_id0, []).append(vtx_id1)
    return result

class Shape:
    '''
    Shape contains a list of points and edge+face connectivities
    '''
    def __init__(self, num_vertices, num_edges=0, num_faces=0):
        self.vertex = np.zeros((num_vertices, 2), dtype=float)
        self.edge = np.zeros((num_edges, 2), dtype=int)
        self.face = np.zeros((num_faces, 3), dtype=int)
        self.position = np.zeros(2)
        self.rotation = 0.0

    def compute_best_transform(self):
        '''
        Returns the 'optimal' position and modify the shape vertices from world space to local space
        Optimal rotation is not computed yet
        '''
        self.position = np.average(self.vertex, axis=0)
        np.subtract(self.vertex, self.position, out=self.vertex)
        self.rotation = 0
        return self.position, self.rotation

    def num_vertices(self):
        return len(self.vertex)

    def num_edges(self):
        return len(self.edge)

    def num_faces(self):
        return len(self.face)
