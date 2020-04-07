"""
@author: Vincent Bonnet
@description : jitted utilities
"""

import numpy as np
import numba
import math

@numba.njit(inline='always')
def triple_product(a, b, c):
    return (a[0] * (b[1]*c[2]-b[2]*c[1]) +
            a[1] * (b[2]*c[0]-b[0]*c[2]) +
            a[2] * (b[0]*c[1]-b[1]*c[0]))

@numba.njit(inline='always')
def isclose(a, b, tol=1.e-8):
    return math.fabs(a - b) < tol

@numba.njit(inline='always')
def cross(a, b):
    result = [a[1]*b[2]-a[2]*b[1],
              a[2]*b[0]-a[0]*b[2],
              a[0]*b[1]-a[1]*b[0]]
    return np.asarray(result)

@numba.njit(inline='always')
def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

@numba.njit(inline='always')
def normalize(v):
    invnorm = 1.0 / math.sqrt(dot(v,v))
    v[0] *= invnorm
    v[1] *= invnorm
    v[2] *= invnorm
