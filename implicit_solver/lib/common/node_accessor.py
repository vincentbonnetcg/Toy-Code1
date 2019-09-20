"""
@author: Vincent Bonnet
@description : This class provides a mapping between node identifiers and datablock layout
Format : [object_id, local_node_id, global_node_id, block_id] #
"""

import numba
import numpy as np

@numba.njit
def empty_node_ids(num_ids):
    return np.empty((num_ids, 4), dtype=np.uint32)

@numba.njit
def emtpy_node_id():
    return np.empty(4, dtype=np.uint32)

@numba.njit
def set_node_id(node_id, object_id = 0, local_node_id = 0, global_node_id = 0, block_id = 0):
    node_id[0] = object_id
    node_id[1] = local_node_id
    node_id[2] = global_node_id
    node_id[3] = block_id

@numba.njit
def node_global_index(node_id):
    return node_id[2]

@numba.njit
def node_xv(dynamics, node_id):
    '''
    dynamics is a tuple of numpy array containing attributes called (x, v)
    '''
    dynamic = dynamics[node_id[0]]
    x = dynamic.data.x[node_id[1]]
    v = dynamic.data.v[node_id[1]]
    return (x, v)

def node_id(dynamic, local_node_id):
    return dynamic.data.node_id[local_node_id]

def node_state(dynamics, node_id):
    dynamic = dynamics[node_id[0]]
    x = dynamic.data.x[node_id[1]]
    v = dynamic.data.v[node_id[1]]
    return (x, v)

def node_add_f(dynamics, node_id, force):
    dynamic = dynamics[node_id[0]]
    dynamic.data.f[node_id[1]] += force
