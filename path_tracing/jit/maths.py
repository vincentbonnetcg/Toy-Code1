"""
@author: Vincent Bonnet
@description : jitted utilities
"""

from . import core as jit_core
import numpy as np
import numba
import math

@numba.njit
def isclose(a, b, tol=1.e-8):
    return math.fabs(a - b) < tol

@numba.njit
def cross(a, b):
    result = [a[1]*b[2]-a[2]*b[1],
              a[2]*b[0]-a[0]*b[2],
              a[0]*b[1]-a[1]*b[0]]
    return np.asarray(result)

@numba.njit
def dot(a, b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

@numba.njit
def ray_triangle(ray_o, ray_d, tv):
    # Moller-Trumbore intersection algorithm
    e1 = tv[1] - tv[0]
    e2 = tv[2] - tv[0]
    ed = ray_o - tv[0]
    tn = cross(e1, e2)
    # explicit linear system (Ax=b) for debugging
    #x = [t, u, v]
    #b = ray_o - tv[0]
    #A = np.zeros((3, 3), dtype=float)
    #A[:,0] = -ray_d
    #A[:,1] = e1

    #A[:,2] = e2
    # solve the system with Cramer's rule
    # det(A) = tripleProduct(-ray_d, e1, e2) = dot(-ray_d, cross(e1,e2))
    detA = dot(-ray_d, tn)
    if isclose(detA, 0.0):
        # ray is parallel to the triangle
        return -1.0

    invDetA = 1.0 / detA

    u = dot(-ray_d, cross(ed, e2)) * invDetA
    if (u < 0.0 or u > 1.0):
        return -1.0

    v = dot(-ray_d, cross(e1, ed)) * invDetA
    if (v < 0.0 or u + v > 1.0):
        return -1.0

    t = dot(ed, tn)  * invDetA
    return t

@numba.njit
def ray_sphere(ray_o, ray_d, sphere_c, sphere_r):
    o = ray_o - sphere_c
    a = dot(ray_d, ray_d)
    b = dot(ray_d, o) * 2.0
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

@numba.njit
def intersect(ray, details):
    min_t = np.finfo(numba.float64).max
    hit = jit_core.Hit(-1.0)
    tri_vertices = details[0]
    tri_normals = details[1]
    tri_materials = details[2]
    # intersection test with triangles
    num_triangles = len(tri_vertices)
    for i in range(num_triangles):
        t = ray_triangle(ray.o, ray.d, tri_vertices[i])
        if t > 0.0 and t < min_t:
            hit.t = t
            hit.p = ray.o + (ray.d * t)
            hit.n = tri_normals[i]
            hit.diffuse = tri_materials[i]
            min_t = t
    # intersection test with sphere
    # TODO

    return hit