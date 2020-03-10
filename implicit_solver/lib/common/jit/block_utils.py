"""
@author: Vincent Bonnet
@description : Datablock jitted utility methods
"""

import numba

@numba.njit
def empty_block_handles():
    return numba.typed.List.empty_list(numba.int32)
