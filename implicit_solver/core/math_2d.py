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

@njit
def distance(x0, x1):
    '''
    Returns distance between x0 and x1
    '''
    distance = norm(x0 - x1)
    return distance

@njit
def area(x0, x1, x2):
    '''
    Returns the area of the 2D triangle from x0, x1, x2
    '''
    u = x1 - x0 # np.subtract(x1, x0)
    v = x2 - x0 # np.subtract(x2, x0)
    area = math.fabs(u[0]*v[1]-v[0]*u[1]) * 0.5
    return area

@njit
def angle(x0, x1, x2):
    '''
    Returns the angle between the segment x0-x1 and x1-x2
    The range is [-pi, pi]
      x1
      /\
     /  \
    x0  x2
    '''
    t01 = x1 - x0
    t01norm = norm(t01)
    t01 /= t01norm
    t12 = x2 - x1
    t12norm =  norm(t12)
    t12 /= t12norm

    # Discrete angle
    det = t01[0]*t12[1] - t01[1]*t12[0]      # determinant
    dot = t01[0]*t12[0] + t01[1]*t12[1]      # dot product
    angle = math.atan2(det,dot)  # atan2 return range [-pi, pi]
    return angle

@njit
def curvature(x0, x1, x2):
    '''
    Connect three points :
      x1
      /\
     /  \
    x0  x2
    Compute the curvature : |dT/ds| where T is the tangent and s the surface
    The curvature at any point along a two-dimensional curve is defined as
    the rate of change in tangent direction θ as a function of arc length s.
    With :
    t01 = x1 - x0 and t12 = x2 - x1
    Discrete curvature formula : angle(t12,t01) / ((norm(t01) + norm(t12)) * 0.5)
    '''
    t01 = x1 - x0
    t01norm = norm(t01)
    t01 /= t01norm
    t12 = x2 - x1
    t12norm =  norm(t12)
    t12 /= t12norm

    # Discrete curvature
    det = t01[0]*t12[1] - t01[1]*t12[0]      # determinant
    dot = t01[0]*t12[0] + t01[1]*t12[1]      # dot product
    angle = math.atan2(det,dot)  # atan2 return range [-pi, pi]
    curvature = angle / ((t01norm + t12norm) * 0.5)

    return curvature
