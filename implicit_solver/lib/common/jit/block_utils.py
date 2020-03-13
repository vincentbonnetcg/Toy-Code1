"""
@author: Vincent Bonnet
@description : Datablock jitted utility methods
"""

import numba
import numpy as np

@numba.njit
def empty_block_handles():
    return numba.typed.List.empty_list(numba.int32)

@numba.njit
def empty_block(block_dtype):
    return np.empty(1, dtype=block_dtype)

@numba.njit
def compute_num_elements(blocks, block_handles = None):
    num_elements = 0

    if block_handles is None:
        for block_container in blocks:
            block_data = block_container[0]
            if block_data['blockInfo_active']:
                num_elements += block_data['blockInfo_numElements']
    else:
        for block_handle in block_handles:
            block_container = blocks[block_handle]
            block_data = block_container[0]
            if block_data['blockInfo_active']:
                num_elements += block_data['blockInfo_numElements']

    return num_elements
