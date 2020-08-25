"""
@author: Vincent Bonnet
@description : Datablock jitted utility methods
"""

import math
import numba
import numpy as np
import lib.common.jit.data_accessor as db
import lib.common.code_gen as generate

@generate.vectorize_block
def set_active(block, active):
    block.blockInfo_active = active

@generate.vectorize_block
def _compute_num_elements(block, ref_counter):
    ref_counter += block.blockInfo_size

@numba.njit
def empty_block_handles():
    return numba.typed.List.empty_list(numba.int32)

@numba.njit
def empty_like_block(blocks):
    block = np.empty_like(blocks[0])
    block[0]['blockInfo_active'] = False
    block[0]['blockInfo_capacity'] = blocks[0][0]['blockInfo_capacity']
    block[0]['blockInfo_size'] = 0
    block[0]['blockInfo_handle'] = -1
    return block

@numba.njit
def get_inactive_block_handles(blocks):
    handles = empty_block_handles()
    for block_index in range(len(blocks)):
        if blocks[block_index][0]['blockInfo_active'] == False:
            handles.append(block_index)
    return handles

@numba.njit
def compute_num_elements(blocks, block_handles = None, func=_compute_num_elements.function):
    counter = np.zeros(1, dtype = np.int32) # use array to pass value as reference
    func(blocks, counter, block_handles)
    return counter[0]

@numba.njit
def init_block(block, block_size, block_handle):
    block[0]['blockInfo_size'] = block_size
    block[0]['blockInfo_active'] = True
    block[0]['blockInfo_handle'] = block_handle

@numba.njit
def init_block_with_ID(block, block_size, block_handle):
    init_block(block, block_size, block_handle)
    data_ID = block[0]['ID']
    for index in range(block_size):
        db.set_data_id(data_ID[index],
                       block_handle,
                       index)

@numba.njit
def append_blocks(blocks, reuse_inactive_block, num_elements, init_func = init_block):
    inactive_block_handles = empty_block_handles()
    block_handles = empty_block_handles()
    block_size = blocks[0][0]['blockInfo_capacity']

    # collect inactive block ids
    if reuse_inactive_block:
        inactive_block_handles = get_inactive_block_handles(blocks)

    # append blocks
    n_blocks = math.ceil(num_elements / block_size)
    for block_index in range(n_blocks):

        block_handle = -1
        block = None

        if reuse_inactive_block and len(inactive_block_handles) > 0:
            # reuse blocks
            block_handle = inactive_block_handles.pop(0)
            block = blocks[block_handle]
        else:
            # allocate a new block
            block_handle = len(blocks)
            block = empty_like_block(blocks)
            blocks.append(block)

        begin_index = block_index * block_size
        block_n_elements = min(block_size, num_elements-begin_index)
        init_func(block, block_n_elements, block_handle)

        # add block id to result
        block_handles.append(block_handle)

    return block_handles
