"""
@author: Vincent Bonnet
@description : This class provides a mapping between node identifiers and datablock layout
Format : [global_node_id, block_handle, block_node_id]
"""

import numba
import numpy as np

ID_SIZE = 2

@numba.njit
def empty_node_ids(num_nodes):
    return np.empty((num_nodes, ID_SIZE), dtype=np.int32)

@numba.njit
def empty_node_id():
    return np.empty(ID_SIZE, dtype=np.int32)

@numba.njit
def set_node_id(node_id, block_handle, block_node_id):
    node_id[0] = block_handle
    node_id[1] = block_node_id

@numba.njit
def node_x(node_blocks, node_id):
    block_handle = node_id[0]
    block_node_id = node_id[1]
    return node_blocks[block_handle][0]['x'][block_node_id]

@numba.njit
def node_v(node_blocks, node_id):
    block_handle = node_id[0]
    block_node_id = node_id[1]
    return node_blocks[block_handle][0]['v'][block_node_id]

@numba.njit
def node_xv(node_blocks, node_id):
    block_handle = node_id[0]
    block_node_id = node_id[1]

    x = node_blocks[block_handle][0]['x'][block_node_id]
    v = node_blocks[block_handle][0]['v'][block_node_id]
    return (x, v)

@numba.njit
def node_add_f(node_blocks, node_id, force):
    block_handle = node_id[0]
    block_node_id = node_id[1]

    f = node_blocks[block_handle][0]['f'][block_node_id]
    f += force

@numba.njit
def node_systemIndex(node_blocks, node_id):
    block_handle = node_id[0]
    block_node_id = node_id[1]
    return node_blocks[block_handle][0]['systemIndex'][block_node_id]

