"""
@author: Vincent Bonnet
@description : jitted utilities
"""

import numpy as np
import numba

def create_polygon_mesh(num_vertices, num_triangles):
    v = np.zeros((num_vertices, 3), dtype=float) # vertices
    t = np.zeros((num_triangles, 3), dtype=int) # triangles
    n = np.zeros((num_triangles, 3), dtype=float) # triangles
    return v, t, n

def create_test_triangle(z_axis):
    v, t, n = create_polygon_mesh(3, 1)
    np.copyto(v[0], [1,0,z_axis])
    np.copyto(v[1], [0,1,z_axis])
    np.copyto(v[2], [-1,0,z_axis])
    np.copyto(t, [0,1,2])
    np.copyto(n[0], [0,0,1])
    return v, t, n

#@numba.njit
def ray_triangle(ray_o, ray_d, tv):
    # Moller-Trumbore intersection algorithm
    e1 = tv[1] - tv[0]
    e2 = tv[2] - tv[0]
    ed = ray_o - tv[0]
    # explicit linear system (Ax=b) for debugging
    #x = [t, u, v]
    #b = ray_o - tv[0]
    #A = np.zeros((3, 3), dtype=float)
    #A[:,0] = -ray_d
    #A[:,1] = e1
    #A[:,2] = e2
    # solve the system with Cramer's rule
    # det(A) = tripleProduct(-ray_d, e1, e2) = dot(-ray_d, cross(e1,e2))
    detA = np.dot(-ray_d, np.cross(e1, e2))
    if np.isclose(detA, 0.0):
        # ray is parallel to the triangle
        return -1.0

    invDetA = 1.0 / detA

    u = np.dot(-ray_d, np.cross(ed, e2)) * invDetA
    if (u < 0.0 or u > 1.0):
        return -1.0

    v = np.dot(-ray_d, np.cross(e1, ed)) * invDetA
    if (v < 0.0 or u + v > 1.0):
        return -1.0

    t = np.dot(ed, np.cross(e1, e2))  * invDetA
    return t

#@numba.njit
def ray_sphere(ray_o, ray_d, sphere_c, sphere_r):
    o = ray_o - sphere_c
    a = np.dot(ray_d, ray_d)
    b = np.dot(ray_d, o) * 2.0
    c = np.dot(o, o) - sphere_r**2
    # solve ax**2 + bx + c = 0
    dis = b**2 - 4*a*c  # discriminant

    if dis < 0.0:
        # no solution
        return -1.0

    if np.isclose(dis, 0.0):
        # one solution
        return -b / 2 * a

    # two solution
    sq = np.sqrt(dis)
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
