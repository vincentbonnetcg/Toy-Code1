"""
@author: Vincent Bonnet
@description : This class provides a mapping between data identifiers and DataBlock layout
Format : [global_node_id, block_handle, block_node_id]'
This file could be auto-generated based on DataBlock
"""

import numba
import numpy as np

ID_SIZE = 2

@numba.njit
def empty_data_ids(num_nodes):
    return np.empty((num_nodes, ID_SIZE), dtype=np.int32)

@numba.njit
def empty_data_id():
    return np.empty(ID_SIZE, dtype=np.int32)

@numba.njit
def set_data_id(ID, block_handle, index):
    ID[0] = block_handle
    ID[1] = index


