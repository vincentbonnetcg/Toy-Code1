"""
@author: Vincent Bonnet
@description : Evaluation of Abstract Syntax Trees
"""

import lib.common.code_gen as generate
import lib.common as common
import numpy as np
import unittest

'''
Datablock Functions
'''
class Vertex:
    def __init__(self):
        self.x = np.ones((2,3)) * 2.1
        self.y = 1.5

class Container:
    def __init__(self, datablock):
        self.data = datablock

def create_datablock():
    datablock = common.DataBlock(Vertex)
    datablock.initialize(10)
    return datablock

'''
Functions to vectorize
'''
@generate.as_vectorized
def add_values(v0 : Vertex, v1 : Vertex, other_value):
    v0.x += v1.x + other_value
    v0.y += v1.y + other_value

'''
CodeGen Tests
'''
class TestCodeGeneration(unittest.TestCase):

    def test_generated_function_with_numpy_input(self):
        datablock0 = create_datablock()
        datablock1 = create_datablock()
        datablock0.lock()
        datablock1.lock()
        add_values(datablock0, datablock1, 1.0)
        self.assertEqual(datablock0.blocks[0]['x'][0][0][0], 5.2)
        self.assertEqual(datablock0.blocks[0]['y'][0], 4.0)

    def test_generated_function_with_datablock_input(self):
        datablock0 = create_datablock()
        datablock1 = create_datablock()
        datablock0.lock()
        datablock1.lock()
        add_values(datablock0, datablock1, 1.0)
        self.assertEqual(datablock0.blocks[0]['x'][0][0][0], 5.2)
        self.assertEqual(datablock0.blocks[0]['y'][0], 4.0)

    def test_function_generated_once(self):
        datablock0 = create_datablock()
        datablock1 = create_datablock()
        datablock0.lock()
        datablock1.lock()
        add_values(datablock0, datablock1, 1.0)
        function0 = add_values.function
        source0 = add_values.source
        add_values(datablock0, datablock1, 1.0)
        function1 = add_values.function
        source1 = add_values.source
        self.assertEqual(function0, function1)
        self.assertEqual(source0, source1)
        self.assertEqual(add_values.njit, True)

    def setUp(self):
        print(" TestCodeGeneration:", self._testMethodName)

if __name__ == '__main__':
    unittest.main(TestCodeGeneration())
