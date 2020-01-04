"""
@author: Vincent Bonnet
@description : Unit tests for datablock
"""

import unittest
import lib.common as common
import numpy as np

'''
Datablock Functions
'''
class ComponentTest:

    def __init__(self):
        self.field_0 = np.float64(0.6)
        self.field_1 = np.ones((2, 2), dtype = np.int64) * 0.5

def create_datablock(num_elements, block_size = 100):
    datablock = common.DataBlock(ComponentTest, block_size)
    datablock.initialize(num_elements)
    return datablock

'''
Datablock Tests
'''
class TestDataBlock(unittest.TestCase):

    def test_datatype(self):
        datablock = create_datablock(num_elements=10)
        datablock_type = np.dtype(datablock.blocks[0])
        self.assertEqual('field_0' in datablock_type.names, True)
        self.assertEqual('field_1' in datablock_type.names, True)
        self.assertEqual('field_c' in datablock_type.names, False)
        self.assertEqual(datablock_type.isalignedstruct, True)
        self.assertEqual(datablock_type.itemsize, 4016)

    def test_default_values(self):
        datablock = create_datablock(num_elements=10)
        block0 = datablock.blocks[0]
        self.assertEqual(block0['field_0'][0], 0.6)
        self.assertTrue((block0['field_1'][0] == [[0.5, 0.5], [0.5, 0.5]]).all())
        self.assertEqual(block0['field_1'][0][0][0], 0.5)

    def test_fill(self):
        datablock = create_datablock(num_elements=10)
        datablock.fill('field_0', 1.5)
        datablock.fill('field_1', 2.5)
        block0 = datablock.blocks[0]
        self.assertEqual(block0['field_0'][0], 1.5)
        self.assertTrue((block0['field_1'][0] == [[2.5, 2.5], [2.5, 2.5]]).all())

    def test_create_blocks(self):
        num_elements = 10
        datablock = create_datablock(num_elements, block_size=3)
        datablock.copyto('field_0', range(num_elements))

        self.assertEqual(len(datablock.blocks), 4)
        self.assertEqual(datablock.blocks[0]['blockInfo_numElements'], 3)
        self.assertEqual(datablock.blocks[3]['blockInfo_numElements'], 1)
        self.assertEqual(datablock.blocks[0]['blockInfo_active'], True)
        self.assertEqual(datablock.blocks[3]['blockInfo_active'], True)
        self.assertTrue((datablock.blocks[0]['field_0'] == [0.,1.,2.]).all())
        self.assertTrue((datablock.blocks[1]['field_0'] == [3.,4.,5.]).all())
        self.assertTrue((datablock.blocks[2]['field_0'] == [6.,7.,8.]).all())
        self.assertTrue((datablock.blocks[3]['field_0'] == [9.,0.6,0.6]).all())

    def test_remove_blocks(self):
        num_elements = 10
        datablock = create_datablock(num_elements, block_size=3)
        datablock.copyto('field_0', range(num_elements))

        datablock.remove([1,2]) # remove block 1 and 2
        flat_array = datablock.flatten('field_0')
        self.assertTrue((flat_array == [0.0,1.0,2.0,9.0]).all())

    def test_inactive_block(self):
        num_elements = 10
        datablock = create_datablock(num_elements, block_size=3)
        datablock.copyto('field_0', range(num_elements))
        datablock.set_active(False, [1,3])
        self.assertEqual(datablock.blocks[0]['blockInfo_active'], True)
        self.assertEqual(datablock.blocks[1]['blockInfo_active'], False)
        self.assertEqual(datablock.blocks[2]['blockInfo_active'], True)
        self.assertEqual(datablock.blocks[3]['blockInfo_active'], False)
        self.assertEqual(datablock.compute_num_elements(), 6)

    def setUp(self):
        print(" TestDataBlock:", self._testMethodName)

if __name__ == '__main__':
    unittest.main(TestDataBlock())
