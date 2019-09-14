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

def create_datablock(num_elements):
    datablock = common.DataBlock()
    datablock.add_field('field_a', np.float64, 1)
    datablock.add_field('field_b', np.float64, (2, 2))
    datablock.add_field_from_class(ComponentTest)
    datablock.initialize(num_elements)
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
        datablock = create_datablock(10)
        datablock_type = np.dtype(datablock.data)
        self.assertEqual('field_a' in datablock_type.names, True)
        self.assertEqual('field_b' in datablock_type.names, True)
        self.assertEqual('field_0' in datablock_type.names, True)
        self.assertEqual('field_1' in datablock_type.names, True)
        self.assertEqual('field_c' in datablock_type.names, False)
        self.assertEqual(datablock_type.isalignedstruct, True)
        self.assertEqual(datablock_type.itemsize, 800)

    def test_datablock_memory_value(self):
        datablock = create_datablock(10)
        self.assertEqual(datablock.field_a[0], 0.0)
        self.assertEqual(datablock.field_0[0], 0.6)
        self.assertEqual(datablock.field_1[0][0][0], 0.5)

    def test_datablock_set_values(self):
        datablock = create_datablock(10)
        set_datablock_values(datablock, 1.5, 2.5)
        self.assertEqual(datablock.field_0[0], 1.5)
        self.assertEqual(datablock.field_1[0][0][0], 2.5)

    def test_blocks(self):
        datablock = create_datablock(10)
        np.copyto(datablock.field_a, range(10))
        blocks, blocks_num_elements = datablock.create_blocks(block_size=3)
        self.assertEqual(len(blocks), 4)
        self.assertTrue((blocks[0]['field_a'] == [0.,1.,2.]).all())
        self.assertTrue((blocks[1]['field_a'] == [3.,4.,5.]).all())
        self.assertTrue((blocks[2]['field_a'] == [6.,7.,8.]).all())
        self.assertTrue((blocks[3]['field_a'] == [9.,0.,0.]).all())

    def setUp(self):
        print(" TestDataBlock:", self._testMethodName)

if __name__ == '__main__':
    unittest.main(TestDataBlock())
