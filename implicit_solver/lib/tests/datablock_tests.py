"""
@author: Vincent Bonnet
@description : Unit tests for datablock
"""

import unittest
import lib.common as common
import numpy as np
from numba import njit, vectorize

'''
Utilities
'''
class ClassType:

    def __init__(self):
        self.field_0 = np.float64(0.6)
        self.field_1 = np.ones((2, 2), dtype = np.int64) * 0.5

def create_test_datablock():
    datablock = common.DataBlock()
    datablock.add_field('field_a', np.float64, 1)
    datablock.add_field('field_b', np.float64, (2, 2))
    datablock.add_field_from_class(ClassType)
    datablock.initialize(10)
    return datablock

'''
Datablock Tests
'''
class TestDataBlock(unittest.TestCase):

    def test_datablock_memory_datatype(self):
        datablock = create_test_datablock()
        datablock_type = np.dtype(datablock.data)
        self.assertEqual('field_a' in datablock_type.names, True)
        self.assertEqual('field_b' in datablock_type.names, True)
        self.assertEqual('field_0' in datablock_type.names, True)
        self.assertEqual('field_1' in datablock_type.names, True)
        self.assertEqual('field_c' in datablock_type.names, False)
        self.assertEqual(datablock_type.isalignedstruct, True)
        self.assertEqual(datablock_type.itemsize, 800)

    def test_datablock_memory_value(self):
        datablock = create_test_datablock()
        datablock.initialize(10)
        self.assertEqual(datablock.field_a[0], 0.0)
        self.assertEqual(datablock.field_1[0][0][0], 0.5)
        self.assertEqual(datablock.field_0[0], 0.6)

    @unittest.skip("skip test_datablock_vectorize")
    def test_datablock_vectorize(self):
        datablock = create_test_datablock()
        # TODO - placeholder for field iterator
        self.assertEqual(True, False)

    def setUp(self):
        print(" TestDataBlock:", self._testMethodName)

if __name__ == '__main__':
    unittest.main()
