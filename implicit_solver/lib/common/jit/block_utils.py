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
    block = np.empty(1, dtype=block_dtype)
    block[0]['blockInfo_active'] = False
    block[0]['blockInfo_numElements'] = 0
    return block

@numba.njit
def inactive_block_handles(blocks):
    handles = empty_block_handles()
    for block_index in range(len(blocks)):
        if blocks[block_index][0]['blockInfo_active'] == False:
            handles.append(block_index)
    return handles

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
