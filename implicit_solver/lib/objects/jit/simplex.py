"""
@author: Vincent Bonnet
@description : Simplices to store point, edge, triangle, tetrahedron
"""

import numpy as np
import lib.common.jit.node_accessor as na

class Point:
    def __init__(self):
        self.x = np.zeros(2, dtype = np.float64)
        self.ID = na.emtpy_node_id()

class Edge:
    def __init__(self):
        self.point_IDs = na.empty_node_ids(2)

class Triangle:
    def __init__(self):
        self.point_IDs = na.empty_node_ids(3)

class Tetrahedron:
    def __init__(self):
        self.point_IDs = na.empty_node_ids(4)

