"""
@author: Vincent Bonnet
@description : Unit tests for datablock
"""

import unittest
import lib.common as common
import numpy as np

class TestDataBlock(unittest.TestCase):

    def test_datablock_memory_datatype(self):
        datablock = common.DataBlock()
        datablock.add_field('field_a', np.float64, 1)
        datablock.add_field('field_b', np.float64, (2, 2))
        datablock.initialize(10)
        datablock_type = np.dtype(datablock.data)
        self.assertEqual('field_a' in datablock_type.names, True)
        self.assertEqual('field_b' in datablock_type.names, True)
        self.assertEqual('field_c' in datablock_type.names, False)
        self.assertEqual(datablock_type.isalignedstruct, True)
        self.assertEqual(datablock_type.itemsize, 400)

    def test_datablock_memory_value(self):
        datablock = common.DataBlock()
        datablock.add_field('field_a', np.float64, 1)
        datablock.initialize(10)
        self.assertEqual(datablock.field_a[0], 0.0)

    def setUp(self):
        print(" TestDataBlock:", self._testMethodName)

if __name__ == '__main__':
    unittest.main()
