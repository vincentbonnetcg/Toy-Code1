"""
@author: Vincent Bonnet
@description : functions to create geometry
"""

import numpy as np

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
