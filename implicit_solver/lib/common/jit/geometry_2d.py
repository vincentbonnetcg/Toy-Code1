"""
@author: Vincent Bonnet
@description : Geometry methods
"""

import numba
import numpy as np
import lib.common.jit.math_2d as math2D
import lib.common.jit.node_accessor as na

parametricSpec = [('points', numba.int32[:,:]), # two points
                  ('t', numba.float32), # parametric value
                  ('position', numba.float64[:]),  # position
                  ('normal', numba.float64[:])]  # normal
@numba.jitclass(parametricSpec)
class ParametricPoint(object):
    def __init__(self):
        self.points = na.empty_node_ids(2)
        self.t = 0.0
        self.position = np.zeros(2, dtype=np.float64)
        self.normal = np.zeros(2, dtype=np.float64)

@numba.njit(inline='always')
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
