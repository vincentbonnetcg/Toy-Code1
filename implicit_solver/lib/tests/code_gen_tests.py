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

def create_datablock():
    datablock = common.DataBlock()
    datablock.add_field_from_class(Vertex)
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
        add_values(datablock0.data, datablock1.data, 1.0)
        self.assertEqual(datablock0.data[0][0][0][0], 5.2)
        self.assertEqual(datablock0.data[1][0], 4.0)

    def test_generated_function_with_datablock_input(self):
        datablock0 = create_datablock()
        datablock1 = create_datablock()
        add_values(datablock0, datablock1, 1.0)
        self.assertEqual(datablock0.data[0][0][0][0], 5.2)
        self.assertEqual(datablock0.data[1][0], 4.0)

    def test_function_generated_once(self):
        datablock0 = create_datablock()
        datablock1 = create_datablock()
        add_values(datablock0.data, datablock1.data, 1.0)
        function0 = add_values.generated_function
        source0 = add_values.generated_source
        add_values(datablock0.data, datablock1.data, 1.0)
        function1 = add_values.generated_function
        source1 = add_values.generated_source
        self.assertEqual(function0, function1)
        self.assertEqual(source0, source1)

    def setUp(self):
        print(" TestCodeGeneration:", self._testMethodName)


if __name__ == '__main__':
    unittest.main(TestCodeGeneration())
