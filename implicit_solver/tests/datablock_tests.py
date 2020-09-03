"""
@author: Vincent Bonnet
@description : Unit tests for datablock
"""

import unittest
import numpy as np
import core
import core.jit.block_utils as block_utils

'''
Datablock Functions
'''
class ComponentTest:

    def __init__(self):
        self.field_0 = np.float64(0.6)
        self.field_1 = np.ones((2, 2), dtype = np.int64) * 0.5

def create_datablock(num_elements, block_size = 100):
    datablock = core.DataBlock(ComponentTest, block_size)
    datablock.initialize(num_elements)
    return datablock

'''
Tests for datablock
'''
class Tests(unittest.TestCase):

    def test_datatype(self):
        datablock = create_datablock(num_elements=10)
        datablock_type = np.dtype(datablock.block(0))
        self.assertEqual('field_0' in datablock_type.names, True)
        self.assertEqual('field_1' in datablock_type.names, True)
        self.assertEqual('field_c' in datablock_type.names, False)
        self.assertEqual(datablock_type.isalignedstruct, True)
        self.assertEqual(datablock_type.itemsize, 4024)

    def test_default_values(self):
        datablock = create_datablock(num_elements=10)
        block0 = datablock.block(0)
        self.assertEqual(block0['field_0'][0], 0.6)
        self.assertTrue((block0['field_1'][0] == [[0.5, 0.5], [0.5, 0.5]]).all())
        self.assertEqual(block0['field_1'][0][0][0], 0.5)

    def test_fill(self):
        datablock = create_datablock(num_elements=10)
        datablock.fill('field_0', 1.5)
        datablock.fill('field_1', 2.5)
        block0 = datablock.block(0)
        self.assertEqual(block0['field_0'][0], 1.5)
        self.assertTrue((block0['field_1'][0] == [[2.5, 2.5], [2.5, 2.5]]).all())

    def test_flatten(self):
        num_elements = 10
        datablock = create_datablock(num_elements, block_size=3)
        field0_data = datablock.flatten('field_0')
        field1_data = datablock.flatten('field_1')
        self.assertTrue((field0_data == 0.6).all())
        self.assertTrue((field1_data == 0.5).all())

    def test_create_blocks(self):
        num_elements = 10
        datablock = create_datablock(num_elements, block_size=3)
        datablock.copyto('field_0', range(num_elements))

        self.assertEqual(len(datablock.blocks), 4)
        self.assertEqual(datablock.block(0)['blockInfo_size'], 3)
        self.assertEqual(datablock.block(3)['blockInfo_size'], 1)
        self.assertEqual(datablock.block(0)['blockInfo_active'], True)
        self.assertEqual(datablock.block(3)['blockInfo_active'], True)
        self.assertEqual(datablock.block(3)['blockInfo_capacity'], 3)
        self.assertEqual(datablock.block(3)['blockInfo_capacity'], 3)
        self.assertTrue((datablock.block(0)['field_0'] == [0.,1.,2.]).all())
        self.assertTrue((datablock.block(1)['field_0'] == [3.,4.,5.]).all())
        self.assertTrue((datablock.block(2)['field_0'] == [6.,7.,8.]).all())
        self.assertTrue((datablock.block(3)['field_0'] == [9.,0.6,0.6]).all())

    def test_inactive_block(self):
        num_elements = 10
        datablock = create_datablock(num_elements, block_size=3)
        datablock.copyto('field_0', range(num_elements))
        block_handles = block_utils.empty_block_handles()
        block_handles.append(1)
        block_handles.append(3)
        block_utils.set_active(datablock, False, block_handles)
        num_elements = block_utils.compute_num_elements(datablock.blocks)
        self.assertEqual(datablock.block(0)['blockInfo_active'], True)
        self.assertEqual(datablock.block(1)['blockInfo_active'], False)
        self.assertEqual(datablock.block(2)['blockInfo_active'], True)
        self.assertEqual(datablock.block(3)['blockInfo_active'], False)
        self.assertEqual(num_elements, 6)

    def test_reuse_inactive_block(self):
        num_elements = 10
        datablock = create_datablock(num_elements, block_size=3)
        self.assertEqual(len(datablock.blocks), 4)

        # disable two blocks [1, 2] and reuse those blocks
        block_handles = block_utils.empty_block_handles()
        block_handles.append(1)
        block_handles.append(2)
        block_utils.set_active(datablock, False, block_handles)
        datablock.append(num_elements = 6, reuse_inactive_block=True)
        self.assertEqual(len(datablock.blocks), 4)

        # check all the blocks are active
        self.assertEqual(datablock.block(0)['blockInfo_active'], True)
        self.assertEqual(datablock.block(1)['blockInfo_active'], True)
        self.assertEqual(datablock.block(2)['blockInfo_active'], True)
        self.assertEqual(datablock.block(3)['blockInfo_active'], True)

    def setUp(self):
        print(" DataBlock Test:", self._testMethodName)

if __name__ == '__main__':
    unittest.main(Tests())
