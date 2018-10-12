"""
@author: Vincent Bonnet
@description : Maths methods
"""
import math
from numba import njit

@njit
def norm(vector):
    '''
    np.linalg.norm is generic and fast for array but pretty slow single for scalar
    Therefore, it is replaced by a less generic norm
    '''
    dot = (vector[0] * vector[0]) + (vector[1] * vector[1])
    return math.sqrt(dot)

@njit
def is_close(a, b, tol=1.e-8):
    '''
    np.isclose is generic and fast for array but pretty slow for single scalar
    Therefore, it is replaced by a less generic norm
    '''
    return math.fabs(a - b) < tol
