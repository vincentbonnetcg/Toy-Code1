"""
@author: Vincent Bonnet
@description : Simplices to store point, edge, triangle, tetrahedron
"""

import numpy as np
import numba # required by lib.common.code_gen

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

@generate.as_vectorized(njit=True,block_handles=True)
def transformPoint(point : Point, rotation_matrix, translate):
    #np.dot(point.x, rotation_matrix, out=point.x) #  not working with Numba0.45.1
    point.x = np.dot(point.x, rotation_matrix)
    point.x += translate
