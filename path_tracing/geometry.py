"""
@author: Vincent Bonnet
@description : functions to create geometry
"""

import numpy as np
from jit import maths as jit_maths

def create_polygon_soup(num_triangles):
    tv = np.zeros((num_triangles, 3, 3), dtype=float) # triangle vertices
    n = np.zeros((num_triangles, 3), dtype=float) # triangle normal
    return tv, n

def create_test_triangle(z_axis):
    tv, n = create_polygon_soup(1)
    np.copyto(tv[0][0], [1,0,z_axis])
    np.copyto(tv[0][1], [0,1,z_axis])
    np.copyto(tv[0][2], [-1,0,z_axis])
    np.copyto(n[0], [0,0,1])
    return tv, n

def create_quad(quad_corners):
    tv, n = create_polygon_soup(2)
    np.copyto(tv[0][0], quad_corners[0])
    np.copyto(tv[0][1], quad_corners[1])
    np.copyto(tv[0][2], quad_corners[2])
    np.copyto(tv[1][0], quad_corners[0])
    np.copyto(tv[1][1], quad_corners[2])
    np.copyto(tv[1][2], quad_corners[3])
    n0 = jit_maths.cross(tv[0][2]-tv[0][0], tv[0][1]-tv[0][0])
    n1 = jit_maths.cross(tv[1][2]-tv[1][0], tv[1][1]-tv[1][0])
    jit_maths.normalize(n0)
    jit_maths.normalize(n1)
    np.copyto(n[0], n0)
    np.copyto(n[1], n1)
    return tv, n
