"""
@author: Vincent Bonnet
@description : Node data type for dynamic object
"""

import numpy as np

class Node:
    '''
    Describes the constraint base
    '''
    def __init__(self):
        self.x = np.zeros(2, dtype = np.float64)
        self.v = np.zeros(2, dtype = np.float64)
        self.f = np.zeros(2, dtype = np.float64)
        self.m = np.float64(0.0)
        self.im = np.float64(0.0)
        # Node id and match result size of node_accessor.node_id() (three unsigned)
        self.node_id = np.zeros(3, dtype=np.uint32)
