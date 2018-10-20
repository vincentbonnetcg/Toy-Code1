"""
@author: Vincent Bonnet
@description : Maths methods
"""
import math
from numba import njit

@njit
def norm(vector):
    '''
    Returns the norm of a 2D vector
    np.linalg.norm is generic and fast for array but pretty slow single for scalar
    Therefore, it is replaced by a less generic norm
    '''
    dot = (vector[0] * vector[0]) + (vector[1] * vector[1])
    return math.sqrt(dot)

@njit
def is_close(value0, value1, tol=1.e-8):
    '''
    Returns whether two scalar are similar
    np.isclose is generic and fast for array but pretty slow for single scalar
    Therefore, it is replaced by a less generic norm
    '''
    return math.fabs(value0 - value1) < tol
