"""
@author: Vincent Bonnet
@description : intersection routines
"""

import math
import numba
import numpy as np
from .maths import dot, isclose, triple_product
from .maths import asub

@numba.njit(inline='always')
def ray_triangle(mempool, tv):
    # Moller-Trumbore intersection algorithm
    asub(tv[1], tv[0], mempool.v[0]) # e1
    asub(tv[2], tv[0], mempool.v[1]) # e2
    asub(mempool.ray_o, tv[0], mempool.v[2]) # ed

    # explicit linear system (Ax=b) for debugging
    #e1 = tv[1] - tv[0]
    #e2 = tv[2] - tv[0]
    #ed = ray_o - tv[0]
    #x = [t, u, v]
    #b = ray_o - tv[0]
    #A = np.zeros((3, 3), dtype=float)
    #A[:,0] = -ray_d
    #A[:,1] = e1
    #A[:,2] = e2
    # solve the system with Cramer's rule
    # det(A) = dot(-ray_d, cross(e1,e2)) = tripleProduct(-ray_d, e1, e2)
    # also det(A) = tripleProduct(ray_d, e1, e2) = -tripleProduct(-ray_d, e1, e2)
    detA = -triple_product(mempool.ray_d, mempool.v[0], mempool.v[1])
    if isclose(detA, 0.0):
        # ray is parallel to the triangle
        return -1.0

    invDetA = 1.0 / detA

    u = -triple_product(mempool.ray_d, mempool.v[2], mempool.v[1]) * invDetA
    if (u < 0.0 or u > 1.0):
        return -1.0

    v = -triple_product(mempool.ray_d, mempool.v[0], mempool.v[2]) * invDetA
    if (v < 0.0 or u + v > 1.0):
        return -1.0

    return triple_product(mempool.v[2], mempool.v[0], mempool.v[1]) * invDetA # t

@numba.njit(inline='always')
def ray_quad(mempool, tv):
    # Moller-Trumbore intersection algorithm
    # same than ray_triangle but different condition on v
    asub(tv[1], tv[0], mempool.v[0]) # e1
    asub(tv[2], tv[0], mempool.v[1]) # e2
    asub(mempool.ray_o, tv[0], mempool.v[2]) # ed

    detA = -triple_product(mempool.ray_d, mempool.v[0], mempool.v[1])
    if isclose(detA, 0.0):
        # ray is parallel to the triangle
        return -1.0

    invDetA = 1.0 / detA

    u = -triple_product(mempool.ray_d, mempool.v[2], mempool.v[1]) * invDetA
    if (u < 0.0 or u > 1.0):
        return -1.0

    v = -triple_product(mempool.ray_d, mempool.v[0], mempool.v[2]) * invDetA
    if (v < 0.0 or v > 1.0):
        return -1.0

    return triple_product(mempool.v[2], mempool.v[0], mempool.v[1]) * invDetA # t

@numba.njit(inline='always')
def ray_sphere(mempool, sphere_c, sphere_r):
    o = mempool.ray_o - sphere_c
    a = dot(mempool.ray_d, mempool.ray_d)
    b = dot(mempool.ray_d, o) * 2.0
    c = dot(o, o) - sphere_r**2
    # solve ax**2 + bx + c = 0
    dis = b**2 - 4*a*c  # discriminant

    if dis < 0.0:
        # no solution
        return -1.0

    if isclose(dis, 0.0):
        # one solution
        return -b / 2 * a

    # two solution
    sq = math.sqrt(dis)
    s1 = (-b-sq) / 2*a  # first solution
    s2 = (-b+sq) / 2*a # second solution

    if s1 < 0.0 and s2 < 0.0:
        return False

    t = s2
    if s1 > 0.0 and s2 > 0.0:
        t = np.minimum(s1, s2)
    elif s1 > 0.0:
        t = s1

    return t