"""
@author: Vincent Bonnet
@description : Simplices to store point, edge, triangle, tetrahedron
"""

import numpy as np
import lib.common.jit.data_accessor as db

class Point:
    def __init__(self):
        self.local_x = np.zeros(2, dtype = np.float64)
        self.x = np.zeros(2, dtype = np.float64)
        self.ID = db.empty_data_id()

class Edge:
    def __init__(self):
        self.point_IDs = db.empty_data_ids(2)
        self.local_normal = np.zeros(2, dtype = np.float64)
        self.normal = np.zeros(2, dtype = np.float64)

class Triangle:
    def __init__(self):
        self.point_IDs = db.empty_data_ids(3)

class Tetrahedron:
    def __init__(self):
        self.point_IDs = db.empty_data_ids(4)


