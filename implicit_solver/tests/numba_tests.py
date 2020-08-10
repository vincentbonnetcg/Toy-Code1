"""
@author: Vincent Bonnet
@description : Unit tests to evaluate Numba capabilities
"""

import numba
import unittest
import numpy as np
import lib.common as common

class ComponentTest:

    def __init__(self):
        self.field_0 = np.float64(0.5)
        self.field_1 = np.ones((2, 2), dtype = np.int64)

def get_block_dtype(block_size = 100):
    datablock = common.DataBlock(ComponentTest, block_size)
    return datablock.dtype_block

@numba.njit
def create_block(block_dtype):
    block_data = np.zeros(1, dtype=block_dtype)
    return block_data

@numba.njit
def set_block(block_data, num_elements, active=True):
    item = block_data[0]
    item.field_0[:] = np.float64(0.5)
    item.field_1[:] = np.ones((2, 2), dtype = np.int64)
    item['blockInfo_size'] = num_elements
    item['blockInfo_active'] = active

@numba.njit
def append_block(array, block_data):
    array.append(block_data)

@numba.njit
def get_inactive_block_indices(array):
    indices = numba.typed.List.empty_list(numba.types.int64)
    for block_index in range(len(array)):
        item = array[block_index][0]
        if not item['blockInfo_active']:
            indices.append(block_index)
    return indices

@numba.njit
def create_typed_list(block_dtype, num_blocks, set_func = set_block):
    array = numba.typed.List()
    # add dummy/inactive block
    block_data = create_block(block_dtype)
    set_func(block_data, num_elements=0, active=False)
    array.append(block_data)

    # add active blocks
    for _ in range(num_blocks):
        block_data = create_block(block_dtype)
        set_func(block_data, num_elements=10, active=True)
        append_block(array, block_data)
    return array

@numba.njit(parallel=False)
def iterate_on_typed_list(array):
    num_blocks = len(array)
    for block_index in numba.prange(num_blocks):
        block_container = array[block_index]
        block_data = block_container[0]
        block_data['blockInfo_size'] = 11

@numba.njit
def take(values, indices = None):
    def add(array, value):
        array[0] += value

    result = np.zeros(1)
    if indices is None:
        for value in values:
            add(result, value)
    else:
        for index in indices:
            add(result, values[index])


    return result[0]

@numba.njit
def test_value_as_reference(refValue, increase):
    def increment(refValue, increase):
        refValue[0] += 10
    increment(refValue, increase)
    return refValue

'''
Waiting for feature on numba => already requested
see : https://github.com/numba/numba/issues/1469
@numba.njit
def zero_shape_array(block_dtype):
    array = numba.typed.List()
    block_data = np.zeros((), dtype=block_dtype)
    array.append(block_data)
    array[0]['field_0'][:] = 0.5
    print(array[0]['field_0'])
    return array
'''
class Tests(unittest.TestCase):

    '''
    def test_zero_shape_array(self):
        block_dtype = get_block_dtype(block_size = 100)
        zero_shape_array(block_dtype)
    '''
    def test_value_as_reference(self):
        value = test_value_as_reference(np.ones(1), 10)
        self.assertEqual(value[0], 11.0)

    def test_typed_list(self):
        block_dtype = get_block_dtype(block_size = 100)
        blocks = create_typed_list(block_dtype, num_blocks = 15)
        self.assertEqual(len(blocks), 16) # include inactive block
        # Test dummy/inactive block
        block_container = blocks[0]
        block_data = block_container[0]
        self.assertEqual(block_data['blockInfo_size'], 0)
        self.assertEqual(block_data['blockInfo_active'], False)

        # Test first active block
        block_container = blocks[1]
        block_data = block_container[0]
        componentTest = ComponentTest()
        self.assertEqual(block_data['blockInfo_size'], 10)
        self.assertEqual(block_data['blockInfo_active'], True)
        self.assertTrue(block_data['field_0'][0] == componentTest.field_0)
        self.assertTrue((block_data['field_1'][0] == componentTest.field_1).all())

    def test_inactive(self):
        block_dtype = get_block_dtype(block_size = 100)
        blocks = create_typed_list(block_dtype, num_blocks = 15)
        block_indices = get_inactive_block_indices(blocks)
        self.assertTrue(len(block_indices) == 1)

    def test_iteration(self):
        block_dtype = get_block_dtype(block_size = 100)
        blocks = create_typed_list(block_dtype, num_blocks = 15)
        iterate_on_typed_list(blocks)
        block_container = blocks[1]
        block_data = block_container[0]
        self.assertEqual(block_data['blockInfo_size'], 11)

    def test_optional_argument(self):
        values = np.ones(10) * 1.5
        indices = np.asarray([1,2,3])
        self.assertEqual(take(values), 15.0)
        self.assertEqual(take(values, indices), 4.5)

    def setUp(self):
        print(" Numba Test:", self._testMethodName)

if __name__ == '__main__':
    unittest.main(Tests())
