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
        self.x = np.zeros(2, dtype = np.float32)
        self.v = np.zeros(2, dtype = np.float32)
        self.f = np.zeros(2, dtype = np.float32)
        self.m = np.float32(0.0)
        self.im = np.float32(0.0)

