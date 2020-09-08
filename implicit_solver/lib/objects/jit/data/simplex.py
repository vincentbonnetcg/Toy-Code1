"""
@author: Vincent Bonnet
@description : Simplices to store point, edge, triangle, tetrahedron
"""

import numpy as np
import core.jit.item_utils as item_utils

class Point:
    def __init__(self):
        self.local_x = np.zeros(2, dtype = np.float64)
        self.x = np.zeros(2, dtype = np.float64)
        self.ID = item_utils.empty_data_id()

    @staticmethod
    def name():
        return "point"

class Edge:
    def __init__(self):
        self.point_IDs = item_utils.empty_data_ids(2)
        self.local_normal = np.zeros(2, dtype = np.float64)
        self.normal = np.zeros(2, dtype = np.float64)

    @staticmethod
    def name():
        return "edge"

class Triangle:
    def __init__(self):
        self.point_IDs = item_utils.empty_data_ids(3)

    @staticmethod
    def name():
        return "triangle"

class Tetrahedron:
    def __init__(self):
        self.point_IDs = item_utils.empty_data_ids(4)

    @staticmethod
    def name():
        return "tetrahedron"

