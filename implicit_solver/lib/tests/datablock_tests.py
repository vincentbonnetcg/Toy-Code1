"""
@author: Vincent Bonnet
@description : Unit tests for datablock
"""

import unittest
import lib.common as common
import numpy as np
from numba import njit, vectorize
from numba_helper import numba_friendly

'''
Datablock Functions
'''
class ComponentTest:

    def __init__(self):
        self.field_0 = np.float64(0.6)
        self.field_1 = np.ones((2, 2), dtype = np.int64) * 0.5

def create_datablock():
    datablock = common.DataBlock()
    datablock.add_field('field_a', np.float64, 1)
    datablock.add_field('field_b', np.float64, (2, 2))
    datablock.add_field_from_class(ComponentTest)
    datablock.initialize(10)
    return datablock

@numba_friendly
@njit
def set_datablock_values(datablock, value0, value1):
    datablock_field_0 = datablock.field_0
    datablock_field_1 = datablock.field_1

    for i in range(len(datablock_field_0)):
        datablock_field_0[i] = value0
        datablock_field_1[i] = value1

'''
Datablock Tests
'''
class TestDataBlock(unittest.TestCase):

    def test_datablock_memory_datatype(self):
        datablock = create_datablock()
        datablock_type = np.dtype(datablock.data)
        self.assertEqual('field_a' in datablock_type.names, True)
        self.assertEqual('field_b' in datablock_type.names, True)
        self.assertEqual('field_0' in datablock_type.names, True)
        self.assertEqual('field_1' in datablock_type.names, True)
        self.assertEqual('field_c' in datablock_type.names, False)
        self.assertEqual(datablock_type.isalignedstruct, True)
        self.assertEqual(datablock_type.itemsize, 800)

    def test_datablock_memory_value(self):
        datablock = create_datablock()
        datablock.initialize(10)
        self.assertEqual(datablock.field_a[0], 0.0)
        self.assertEqual(datablock.field_0[0], 0.6)
        self.assertEqual(datablock.field_1[0][0][0], 0.5)

    def test_datablock_set_values(self):
        datablock = create_datablock()
        datablock.initialize(10)
        set_datablock_values(datablock, 1.5, 2.5)
        self.assertEqual(datablock.field_0[0], 1.5)
        self.assertEqual(datablock.field_1[0][0][0], 2.5)

    @unittest.skip("skip test_datablock_vectorize")
    def test_datablock_vectorize(self):
        datablock = create_datablock()
        # TODO - placeholder for field iterator
        self.assertEqual(True, False)

    def setUp(self):
        print(" TestDataBlock:", self._testMethodName)

if __name__ == '__main__':
    unittest.main(TestDataBlock())
