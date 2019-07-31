"""
@author: Vincent Bonnet
@description : Evaluation of Abstract Syntax Trees
"""

import lib.common.code_gen as generate
import lib.common as common
import numpy as np

'''
Datablock Functions
'''
class Vertex:
    def __init__(self):
        self.x = np.ones((2,3)) * 2.1
        self.y = 1.5

def create_datablock():
    datablock = common.DataBlock()
    datablock.add_field_from_class(Vertex)
    datablock.initialize(10)
    return datablock

'''
Functions to vectorize
'''
@generate.as_vectorized
def test_add(v0, v1, other_value):
    v0.x += v1.x + other_value
    v0.y += v1.y + other_value


'''
Test
'''
datablock0 = create_datablock()
datablock1 = create_datablock()
test_add(datablock0.data, datablock1.data, 1.0)
print(datablock0.data)