"""
@author: Vincent Bonnet
@description : Shape description
"""

import numpy as np
import lib.common.jit.math_2d as math2D

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

    def transform(self,position, rotation):
        mat = math2D.rotation_matrix(rotation)
        self.vertex = np.dot(self.vertex, mat)
        self.vertex += np.asarray(position)

    def num_vertices(self):
        return len(self.vertex)

    def num_edges(self):
        return len(self.edge)

    def num_faces(self):
        return len(self.face)

    def get_edge_surface_data(self):
        '''
        Returns the edge ids on the surface and associated normals
        '''
        edges_map = dict()

        for face_vtx in self.face:
            for i in range(3): # loop around the edges
                vtx0 = face_vtx[i]
                vtx1 = face_vtx[(i+1)%3]
                vtx2 = face_vtx[(i+2)%3]

                # create a edge_key
                edge_key = tuple([vtx0,vtx1])
                if edge_key[0] > edge_key[1]:
                    edge_key = tuple([vtx1,vtx0])

                # add normal
                edge1_dir = self.vertex[vtx1] - self.vertex[vtx0]
                edge2_dir = self.vertex[vtx2] - self.vertex[vtx0]
                edge1_dir /= math2D.norm(edge1_dir)
                edge_normal = np.asarray([-edge1_dir[1], edge1_dir[0]])
                if math2D.dot(edge2_dir, edge_normal) > 0:
                    edge_normal = np.asarray([edge1_dir[1], -edge1_dir[0]])

                info = edges_map.get(edge_key, [0, (0.0, 0.0)])
                info[0] += 1
                info[1] = edge_normal
                edges_map[edge_key] = info

        # retrieve the surface edges
        surface_edge_normals = []
        surface_edge_ids = []

        for k, v in edges_map.items():
            if v[0] == 1:
                surface_edge_normals.append(v[1])
                surface_edge_ids.append(k)

        return np.asarray(surface_edge_ids), np.asarray(surface_edge_normals)

