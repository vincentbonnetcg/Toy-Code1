"""
@author: Vincent Bonnet
@description : This class provides a mapping between node identifiers and data
"""

import numba

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
