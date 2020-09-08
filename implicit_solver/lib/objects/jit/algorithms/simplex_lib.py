"""
@author: Vincent Bonnet
@description : simplex classes and methods
"""

import numpy as np
import numba

from lib.objects.jit.data import Point, Edge, Triangle
import core.code_gen as generate
import core.jit.item_utils as item_utils
import lib.objects.jit.algorithms.data_accessor as db
import core.jit.math_2d as math2D

closestResultSpec = [('points', numba.int32[:,:]), # two points
                  ('t', numba.float32), # parametric value
                  ('position', numba.float64[:]),  # position
                  ('normal', numba.float64[:]),# normal
                  ('squared_distance', numba.float64)] # parametric value
@numba.experimental.jitclass(closestResultSpec)
class ClosestResult(object):
    def __init__(self):
        self.points = item_utils.empty_data_ids(2)
        self.t = 0.0
        self.position = np.zeros(2, dtype=np.float64)
        self.normal = np.zeros(2, dtype=np.float64)
        self.squared_distance = np.finfo(np.float64).max

insideResultSpec = [('isInside', numba.boolean)]
@numba.experimental.jitclass(insideResultSpec)
class IsInsideResult(object):
    def __init__(self):
        self.isInside = False

@generate.vectorize
def transform_point(point : Point, rotation_matrix, translate):
    #np.dot(point.x, rotation_matrix, out=point.x) #  not working with Numba0.45.1
    point.x = np.dot(point.local_x, rotation_matrix)
    point.x += translate

@generate.vectorize
def transform_normal(edge : Edge, rotation_matrix):
    edge.normal = np.dot(edge.local_normal, rotation_matrix)

@generate.vectorize
def get_closest_param(edge : Edge, points, position, o_param):
    # o_param = ClosestResult()
    x0 = db.x(points, edge.point_IDs[0])
    x1 = db.x(points, edge.point_IDs[1])

    edge_dir = x1 - x0 # could be precomputed
    edge_dir_square = math2D.dot(edge_dir, edge_dir) # could be precomputed
    proj_p = math2D.dot(position - x0, edge_dir)
    t = proj_p / edge_dir_square
    t = max(min(t, 1.0), 0.0)
    projected_point = x0 + edge_dir * t # correct the project point
    vector_distance = (position - projected_point)
    squared_distance = math2D.dot(vector_distance, vector_distance)
    # update the minimum distance
    if squared_distance < o_param.squared_distance:
        o_param.points = edge.point_IDs
        o_param.t = t
        o_param.squared_distance = squared_distance
        o_param.position = x0 * (1.0 - t) + x1 * t
        o_param.normal = edge.normal

@generate.vectorize
def is_inside(face : Triangle, points, position, o_result):
    # result = IsInsideResult()
    x0 = db.x(points, face.point_IDs[0])
    x1 = db.x(points, face.point_IDs[1])
    x2 = db.x(points, face.point_IDs[2])
    v0 = x2 - x0
    v1 = x1 - x0
    v2 = position - x0

    dot00 = math2D.dot(v0, v0)
    dot01 = math2D.dot(v0, v1)
    dot02 = math2D.dot(v0, v2)
    dot11 = math2D.dot(v1, v1)
    dot12 = math2D.dot(v1, v2)

    inv = 1.0 / (dot00 * dot11 - dot01 * dot01)
    a = (dot11 * dot02 - dot01 * dot12) * inv
    b = (dot00 * dot12 - dot01 * dot02) * inv
    if a>=0 and b>=0 and a+b<=1:
        o_result.isInside = True
        return