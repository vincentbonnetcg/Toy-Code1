"""
@author: Vincent Bonnet
@description : This class provides a mapping between node identifiers and datablock layout
Format : [object_id, object_node_id, global_node_id, block_id, block_node_id] #
"""

import numba
import numpy as np

@numba.njit
def empty_node_ids(num_ids):
    return np.empty((num_ids, 5), dtype=np.uint32)

@numba.njit
def emtpy_node_id():
    return np.empty(5, dtype=np.uint32)

@numba.njit
def set_node_id(node_id, object_id = 0, object_node_id = 0, global_node_id = 0, block_id = 0, block_node_id = 0):
    node_id[0] = object_id
    node_id[1] = object_node_id
    node_id[2] = global_node_id
    node_id[3] = block_id
    node_id[4] = block_node_id

@numba.njit
def node_global_index(node_id):
    return node_id[2]

def node_xv(dynamics, node_id):
    object_id = node_id[0]
    block_id = node_id[3]
    block_node_id = node_id[4]

    dynamic = dynamics[object_id]
    x = dynamic.data.blocks[block_id]['x'][block_node_id]
    v = dynamic.data.blocks[block_id]['v'][block_node_id]
    return (x, v)

def node_add_f(dynamics, node_id, force):
    dynamic = dynamics[node_id[0]]
    dynamic.data.f[node_id[1]] += force

def get_node_id_from_vertex_id(dynamic, vertex_id):
    '''
    Temporary - slow method
    Map node id and vertex id
    '''
    data_node_id = dynamic.data.flatten('node_id')
    for node_id in data_node_id:
        if node_id[1] == vertex_id:
            return node_id
