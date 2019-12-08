"""
@author: Vincent Bonnet
@description : This class provides a mapping between node identifiers and datablock layout
Format : [object_id, object_node_id, global_node_id, block_id, block_node_id] #
"""

import numba
import numpy as np

ID_SIZE = 4

@numba.njit
def empty_node_ids(num_nodes):
    return np.empty((num_nodes, ID_SIZE), dtype=np.uint32)

@numba.njit
def emtpy_node_id():
    return np.empty(ID_SIZE, dtype=np.uint32)

@numba.njit
def set_node_id(node_id, object_id, global_node_id, block_id, block_node_id):
    node_id[0] = object_id
    node_id[1] = global_node_id
    node_id[2] = block_id
    node_id[3] = block_node_id

@numba.njit
def set_object_id(node_id, object_id, global_node_id):
    node_id[0] = object_id
    node_id[1] = global_node_id

@numba.njit
def node_global_index(node_id):
    return node_id[1]

def node_x(dynamics, node_id):
    object_id = node_id[0]
    block_id = node_id[2]
    block_node_id = node_id[3]
    return dynamics[object_id].data.blocks[block_id]['x'][block_node_id]

def node_v(dynamics, node_id):
    object_id = node_id[0]
    block_id = node_id[2]
    block_node_id = node_id[3]
    return dynamics[object_id].data.blocks[block_id]['v'][block_node_id]

def node_xv(dynamics, node_id):
    object_id = node_id[0]
    block_id = node_id[2]
    block_node_id = node_id[3]

    dynamic = dynamics[object_id]
    x = dynamic.data.blocks[block_id]['x'][block_node_id]
    v = dynamic.data.blocks[block_id]['v'][block_node_id]
    return (x, v)

def node_add_f(dynamics, node_id, force):
    object_id = node_id[0]
    block_id = node_id[2]
    block_node_id = node_id[3]

    dynamic = dynamics[object_id]
    f = dynamic.data.blocks[block_id]['f'][block_node_id]
    f += force
