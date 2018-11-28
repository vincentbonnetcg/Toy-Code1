"""
@author: Vincent Bonnet
@description : Maths methods
"""
import math
from numba import njit

@njit
def dot(v0, v1):
    '''
    Returns the dotproduct of a 2D vector
    np.dot is generic and fast for array but slow for a single for scalar
    Therefore, it is replaced by a less generic norm
    '''
    return (v0[0] * v1[0]) + (v0[1] * v1[1])

@njit
def norm(vector):
    '''
    Returns the norm of a 2D vector
    np.linalg.norm is generic and fast for array but slow for a single for scalar
    Therefore, it is replaced by a less generic norm
    '''
    dot = (vector[0] * vector[0]) + (vector[1] * vector[1])
    return math.sqrt(dot)

@njit
def is_close(value0, value1, tol=1.e-8):
    '''
    Returns whether two scalar are similar
    np.isclose is generic and fast for array but slow for a single for scalar
    Therefore, it is replaced by a less generic norm
    '''
    return math.fabs(value0 - value1) < tol
