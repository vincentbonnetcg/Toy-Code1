"""
@author: Vincent Bonnet
@description : Node data type for dynamic object
"""

import numpy as np
import lib.common.jit.node_accessor as na

class Node:
    def __init__(self):
        self.x = np.zeros(2, dtype = np.float64)
        self.v = np.zeros(2, dtype = np.float64)
        self.f = np.zeros(2, dtype = np.float64)
        self.m = np.float64(0.0)
        self.im = np.float64(0.0)
        self.ID = na.empty_node_id()
