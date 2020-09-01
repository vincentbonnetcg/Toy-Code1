"""
@author: Vincent Bonnet
@description : Kinematic object describes animated objects
"""

import numpy as np
from lib.common import Shape

class Kinematic:
    '''
    Kinematic describes an animated object
    '''
    def __init__(self, details, shape):
        # append points
        db_points = details.db['point']
        self.point_handles =  db_points.append_empty(len(shape.vertex))
        db_points.copyto('local_x', shape.vertex, self.point_handles)
        db_points.copyto('x', shape.vertex, self.point_handles)
        point_ids = db_points.flatten('ID', self.point_handles)
        # append edges
        db_edges = details.db['edge']
        surface_edge_ids, surface_edge_normals = shape.get_edge_surface_data()
        edge_pids = np.take(point_ids, surface_edge_ids, axis=0)
        self.edge_handles = db_edges.append_empty(len(surface_edge_ids))
        db_edges.copyto('local_normal', surface_edge_normals, self.edge_handles)
        db_edges.copyto('normal', surface_edge_normals, self.edge_handles)
        db_edges.copyto('point_IDs', edge_pids, self.edge_handles)
        # append triangles
        self.face_ids = np.copy(shape.face)
        triangle_pids = np.take(point_ids, self.face_ids, axis=0)
        db_triangles = details.db['triangle']
        self.triangle_handles = db_triangles.append_empty(len(self.face_ids))
        db_triangles.copyto('point_IDs', triangle_pids, self.triangle_handles)
        # metadata
        self.meta_data = {}

    def get_as_shape(self, details):
        x = details.db['point'].flatten('x', self.point_handles)
        shape = Shape(len(x), 0, len(self.face_ids))
        np.copyto(shape.vertex, x)
        np.copyto(shape.face, self.face_ids)
        return shape

    def metadata(self):
        return self.meta_data
