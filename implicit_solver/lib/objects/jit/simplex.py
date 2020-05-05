"""
@author: Vincent Bonnet
@description : Simplices to store point, edge, triangle, tetrahedron
"""

import numpy as np
import numba # required by lib.common.code_gen

import lib.common.jit.math_2d as math2D
import lib.common.jit.node_accessor as na
import lib.common.code_gen as generate

class Point:
    def __init__(self):
        self.x = np.zeros(2, dtype = np.float64)
        self.ID = na.emtpy_node_id()

class Edge:
    def __init__(self):
        self.point_IDs = na.empty_node_ids(2)
        self.normal = np.zeros(2, dtype = np.float64)

class Triangle:
    def __init__(self):
        self.point_IDs = na.empty_node_ids(3)

class Tetrahedron:
    def __init__(self):
        self.point_IDs = na.empty_node_ids(4)

@generate.as_vectorized(njit=True, block_handles=True)
def transformPoint(point : Point, rotation_matrix, translate):
    #np.dot(point.x, rotation_matrix, out=point.x) #  not working with Numba0.45.1
    point.x = np.dot(point.x, rotation_matrix)
    point.x += translate

@generate.as_vectorized(njit=True, block_handles=True)
def get_closest_param(edge : Edge, points, position, o_param):
    # o_param = ClosestResult()
    x0 = na.node_x(points, edge.point_IDs[0])
    x1 = na.node_x(points, edge.point_IDs[1])

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
