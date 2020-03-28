"""
@author: Vincent Bonnet
@description : Geometry methods
"""

import numba
import numpy as np
import lib.common.jit.math_2d as math2D

parametricSpec = [('index', numba.int32), # simplex index
                  ('t', numba.float32), # parametric value
                  ('position', numba.float64[:]),  # position
                  ('normal', numba.float64[:])]  # normal

@numba.jitclass(parametricSpec)
class ParametricPoint(object):
    def __init__(self, index, t):
        self.index = index
        self.t = t
        self.position = np.zeros(2, dtype=np.float64)
        self.normal = np.zeros(2, dtype=np.float64)

@numba.njit
def is_inside(point, vertices, face_ids):
    for i in range(len(face_ids)):
        edge_vtx = [vertices[face_ids[i][0]],
                    vertices[face_ids[i][1]],
                    vertices[face_ids[i][2]]]

        v0 = edge_vtx[2] - edge_vtx[0]
        v1 = edge_vtx[1] - edge_vtx[0]
        v2 = point - edge_vtx[0]

        dot00 = math2D.dot(v0, v0)
        dot01 = math2D.dot(v0, v1)
        dot02 = math2D.dot(v0, v2)
        dot11 = math2D.dot(v1, v1)
        dot12 = math2D.dot(v1, v2)

        inv = 1.0 / (dot00 * dot11 - dot01 * dot01)
        a = (dot11 * dot02 - dot01 * dot12) * inv
        b = (dot00 * dot12 - dot01 * dot02) * inv

        if a>=0 and b>=0 and a+b<=1:
            return True

    return False

@numba.njit
def get_closest_position(point, vertices, edge_ids, edge_normals):
    param = ParametricPoint(-1, 0.0)
    min_distance2 = np.finfo(np.float64).max

    for i in range(len(edge_ids)):
        edge_vtx = [vertices[edge_ids[i][0]],
                    vertices[edge_ids[i][1]]]

        edge_dir = edge_vtx[1] - edge_vtx[0] # could be precomputed
        edge_dir_square = math2D.dot(edge_dir, edge_dir) # could be precomputed
        proj_p = math2D.dot(point - edge_vtx[0], edge_dir)
        t = proj_p / edge_dir_square
        t = max(min(t, 1.0), 0.0)
        projected_point = edge_vtx[0] + edge_dir * t # correct the project point
        vector_distance = (point - projected_point)
        distance2 = math2D.dot(vector_distance, vector_distance)
        # update the minimum distance
        if distance2 < min_distance2:
            param.index = i
            param.t = t
            min_distance2 = distance2

    # set position
    v0 = edge_ids[param.index][0]
    v1 = edge_ids[param.index][1]
    param.position = vertices[v0] * (1.0 - param.t) + vertices[v1] * param.t

    # set normal
    param.normal = edge_normals[param.index]

    return param
