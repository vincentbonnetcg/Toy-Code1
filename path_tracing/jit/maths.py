"""
@author: Vincent Bonnet
@description : jitted utilities
"""

import numpy as np
import numba
import math

@numba.njit(inline='always')
def asub(a, b, out):
    # squeeze some performance by skipping the generic np.subtract
    out[0] = a[0] - b[0]
    out[1] = a[1] - b[1]
    out[2] = a[2] - b[2]

@numba.njit(inline='always')
def axpy(a, x, y, out):
    out[0] = y[0] + (x[0] * a)
    out[1] = y[1] + (x[1] * a)
    out[2] = y[2] + (x[2] * a)

@numba.njit(inline='always')
def copy(x, y):
    x[0] = y[0]
    x[1] = y[1]
    x[2] = y[2]

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

@numba.njit(inline='always')
def compute_tangent(n):
    tangent = [0.0,0.0,0.0]
    if abs(n[0]) > abs(n[1]):
        ntdot = n[0]**2+n[2]**2
        tangent[0] = n[2]/ntdot
        tangent[2] = -n[0]/ntdot
    else:
        ntdot = n[1]**2+n[2]**2
        tangent[1] = -n[2]/ntdot
        tangent[2] = n[1]/ntdot
    return np.asarray(tangent)

@numba.njit(inline='always')
def compute_tangents_binormals(normals, tangents, binormals):
    for i in range(len(normals)):
        tangents[i] = compute_tangent(normals[i])
        binormals[i] = cross(normals[i], tangents[i])
