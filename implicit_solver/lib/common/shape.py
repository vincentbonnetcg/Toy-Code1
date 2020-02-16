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
    Shape contains a flat list of vertices and connectivities (edge,face)
    '''
    def __init__(self, num_vertices, num_edges=0, num_faces=0):
        self.vertex = np.zeros((num_vertices, 2), dtype=float)
        self.edge = np.zeros((num_edges, 2), dtype=int)
        self.face = np.zeros((num_faces, 3), dtype=int)

    def compute_best_transform(self):
        '''
        Returns the 'optimal' position and modify the shape vertices from world space to local space
        Optimal rotation is not computed yet
        '''
        position = np.average(self.vertex, axis=0)
        np.subtract(self.vertex, position, out=self.vertex)
        rotation = 0.0

        return position, rotation

    def num_vertices(self):
        return len(self.vertex)

    def num_edges(self):
        return len(self.edge)

    def num_faces(self):
        return len(self.face)

    def get_edges_on_surface(self):
        edges_map = dict()

        for face_vtx in self.face:
            for i in range(3):
                vtx0 = face_vtx[i]
                vtx1 = face_vtx[(i+1)%3]

                # create a edge_key
                edge_key = tuple([vtx0,vtx1])
                if edge_key[0] > edge_key[1]:
                    edge_key = tuple([vtx1,vtx0])

                # add coutner on the edge
                counter = edges_map.get(edge_key, 0)
                edges_map[edge_key] = counter+1

        # retrieve the surface edges
        surface_edges = []
        for k, v in edges_map.items():
            if v == 1:
                surface_edges.append(k)

        return np.asarray(surface_edges)


