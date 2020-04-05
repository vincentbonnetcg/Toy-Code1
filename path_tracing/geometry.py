"""
@author: Vincent Bonnet
@description : functions to create geometry
"""

import numpy as np
from jit import maths as jit_maths

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

def create_quadrilateral(v0, v1, v2, v3):
    v, t, n = create_polygon_mesh(4, 2)
    np.copyto(v[0], v0)
    np.copyto(v[1], v1)
    np.copyto(v[2], v2)
    np.copyto(v[3], v3)
    np.copyto(t[0], [0,1,2])
    np.copyto(t[1], [0,2,3])
    n0 = jit_maths.cross(v[2]-v[0], v[1]-v[0])
    n1 = jit_maths.cross(v[3]-v[0], v[2]-v[0])
    jit_maths.normalize(n0)
    jit_maths.normalize(n1)
    np.copyto(n[0], n0)
    np.copyto(n[1], n1)
    return v, t, n

