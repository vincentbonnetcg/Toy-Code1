"""
@author: Vincent Bonnet
@description : Maths methods
"""
import math

def norm(vector):
    '''
    np.linalg.norm is generic and fast for array but pretty slow single scalar
    It is therefore replaced by a less generic norm
    '''
    dot = (vector[0] * vector[0]) + (vector[1] * vector[1])
    return math.sqrt(dot)

