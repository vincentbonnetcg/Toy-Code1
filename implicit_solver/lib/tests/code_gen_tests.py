"""
@author: Vincent Bonnet
@description : Evaluation of Abstract Syntax Trees
"""

import inspect
import lib.common as common
import numpy as np
from numba_helper import numba_friendly
from numba import njit, vectorize

'''
Datablock Functions
'''
class Vertex:
    def __init__(self):
        self.x = 1.2
        self.y = 1.5

def create_datablock():
    datablock = common.DataBlock()
    datablock.add_field_from_class(Vertex)
    datablock.initialize(10)
    return datablock

'''
Functions
'''
def test_add(v0 : Vertex, v1 : Vertex, other_value = 1.0):
    v0.x += v1.x + other_value
    v0.y += v1.y + other_value

@numba_friendly
def DB_test_add(array_v0, array_v1):
    '''
    array_v0 contains {x, y}
    '''
    v0_x = array_v0['x']
    v0_y = array_v0['y']
    v1_x = array_v1['x']
    v1_y = array_v1['y']

    for i in range(v0_x.shape[0]):
        v0_x[i] += v1_x[i]
        v0_y[i] += v1_y[i]

'''
Test
'''
# TODO : generate DB_test_add(...) from test_add(...)

datablock0 = create_datablock()
datablock1 = create_datablock()
DB_test_add(datablock0, datablock1)
print(datablock0.x)
print(datablock0.y)

function_args = inspect.signature(test_add)
for key in function_args.parameters:
    print(key)
