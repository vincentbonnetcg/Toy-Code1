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
def node_global_index(node_id):
    return node_id[1]

def node_x(details_node, node_id):
    block_id = node_id[2]
    block_node_id = node_id[3]
    return details_node.blocks[block_id]['x'][block_node_id]

def node_v(details_node, node_id):
    block_id = node_id[2]
    block_node_id = node_id[3]
    return details_node.blocks[block_id]['v'][block_node_id]

def node_xv(details_node, node_id):
    block_id = node_id[2]
    block_node_id = node_id[3]

    x = details_node.blocks[block_id]['x'][block_node_id]
    v = details_node.blocks[block_id]['v'][block_node_id]
    return (x, v)

def node_add_f(details_node, node_id, force):
    block_id = node_id[2]
    block_node_id = node_id[3]

    f = details_node.blocks[block_id]['f'][block_node_id]
    f += force
