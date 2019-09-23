"""
@author: Vincent Bonnet
@description : Unit tests for datablock
"""

import unittest
import lib.common as common
import numpy as np
from numba import njit

'''
Datablock Functions
'''
class ComponentTest:

    def __init__(self):
        self.field_0 = np.float64(0.6)
        self.field_1 = np.ones((2, 2), dtype = np.int64) * 0.5

def create_datablock(num_elements, block_size = 100):
    datablock = common.DataBlock(block_size)
    datablock.add_field('field_a', np.float64, 1)
    datablock.add_field('field_b', np.float64, (2, 2))
    datablock.add_field_from_class(ComponentTest)
    datablock.initialize(num_elements)
    return datablock

'''
Datablock Tests
'''
class TestDataBlock(unittest.TestCase):

    def test_datablock_datatype(self):
        datablock = create_datablock(num_elements=10)
        datablock_type = np.dtype(datablock.blocks[0])
        self.assertEqual('field_a' in datablock_type.names, True)
        self.assertEqual('field_b' in datablock_type.names, True)
        self.assertEqual('field_0' in datablock_type.names, True)
        self.assertEqual('field_1' in datablock_type.names, True)
        self.assertEqual('field_c' in datablock_type.names, False)
        self.assertEqual(datablock_type.isalignedstruct, True)
        self.assertEqual(datablock_type.itemsize, 8008)

    def test_datablock_default_values(self):
        datablock = create_datablock(num_elements=10)
        block0 = datablock.blocks[0]
        self.assertEqual(block0['field_a'][0], 0.0)
        self.assertEqual(block0['field_0'][0], 0.6)
        self.assertTrue((block0['field_1'][0] == [[0.5, 0.5], [0.5, 0.5]]).all())

    def test_datablock_set_values(self):
        datablock = create_datablock(num_elements=10)
        datablock.fill('field_0', 1.5)
        datablock.fill('field_1', 2.5)
        block0 = datablock.blocks[0]
        self.assertEqual(block0['field_0'][0], 1.5)
        self.assertTrue((block0['field_1'][0] == [[2.5, 2.5], [2.5, 2.5]]).all())

    def test_blocks(self):
        num_elements = 10
        datablock = create_datablock(num_elements, block_size=3)
        datablock.copyto('field_a', range(num_elements))

        self.assertEqual(len(datablock.blocks), 4)
        self.assertEqual(datablock.blocks[0]['blockInfo_numElements'], 3)
        self.assertEqual(datablock.blocks[3]['blockInfo_numElements'], 1)
        self.assertTrue((datablock.blocks[0]['field_a'] == [0.,1.,2.]).all())
        self.assertTrue((datablock.blocks[1]['field_a'] == [3.,4.,5.]).all())
        self.assertTrue((datablock.blocks[2]['field_a'] == [6.,7.,8.]).all())
        self.assertTrue((datablock.blocks[3]['field_a'] == [9.,0.,0.]).all())

    def test_blocks_default_values(self):
        num_elements = 10
        datablock = create_datablock(num_elements, block_size=3)
        block0 = datablock.blocks[0]
        self.assertEqual(block0['field_a'][0], 0.0)
        self.assertEqual(block0['field_0'][0], 0.6)
        self.assertEqual(block0['field_1'][0][0][0], 0.5)

    def setUp(self):
        print(" TestDataBlock:", self._testMethodName)

if __name__ == '__main__':
    unittest.main(TestDataBlock())
