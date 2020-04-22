"""
@author: Vincent Bonnet
@description : basic render routines for mix data pathtracer (triangle/quad/sphere)
"""

import numba
import numpy as np
from .maths import cross, dot, compute_tangent
from .maths import copy, axpy
from . import intersect

@numba.njit
def ray_details(details, mempool, skip_face_id = -1):
    # details from Scene.mix_details()
    mempool.next_hit() # use the next allocated hit
    min_t = np.finfo(numba.float64).max
    quad_vertices = details[0]
    quad_normals = details[1]
    quad_tangents = details[2]
    quad_binormals = details[3]
    quad_materials = details[4]
    tri_vertices = details[5]
    tri_normals = details[6]
    tri_tangents = details[7]
    tri_binormals = details[8]
    tri_materials = details[9]
    sphere_params = details[10]
    sphere_materials = details[11]
    hit_type = -1
    hit_id = -1
    # intersection test with triangles
    num_quads = len(quad_vertices)
    for i in range(num_quads):
        if i == skip_face_id:
            continue
        t = intersect.ray_quad(mempool, quad_vertices[i])
        mempool.total_intersection += 1
        if t > 0.0 and t < min_t:
            min_t = t
            hit_type = 0
            hit_id = i

    # intersection test with triangles
    num_triangles = len(tri_vertices)
    for i in range(num_triangles):
        if i == skip_face_id:
            continue
        t = intersect.ray_triangle(mempool, tri_vertices[i])
        mempool.total_intersection += 1
        if t > 0.0 and t < min_t:
            min_t = t
            hit_type = 1
            hit_id = i

    # intersection test with spheres
    num_spheres = len(sphere_params)
    for i in range(num_spheres):
        c = sphere_params[i].c
        r = sphere_params[i].r
        t = intersect.ray_sphere(mempool, c, r)
        mempool.total_intersection += 1
        if t > 0.0 and t < min_t:
            min_t = t
            hit_type = 2
            hit_id = i

    if hit_type == 0: # quad hit
        i = mempool.depth
        mempool.hit_t[i] = min_t
        axpy(min_t, mempool.ray_d, mempool.ray_o, mempool.hit_p[i])
        copy(mempool.hit_n[i], quad_normals[hit_id])
        copy(mempool.hit_tn[i], quad_tangents[hit_id])
        copy(mempool.hit_bn[i], quad_binormals[hit_id])
        mempool.hit_face_id[i], hit_id
        copy(mempool.hit_reflectance[i], quad_materials[hit_id][0])
        copy(mempool.hit_emittance[i], quad_materials[hit_id][1])
    elif hit_type == 1: # triangle hit
        i = mempool.depth
        mempool.hit_t[i] = min_t
        axpy(min_t, mempool.ray_d, mempool.ray_o, mempool.hit_p[i])
        copy(mempool.hit_n[i], tri_normals[hit_id])
        copy(mempool.hit_tn[i], tri_tangents[hit_id])
        copy(mempool.hit_bn[i], tri_binormals[hit_id])
        mempool.hit_face_id[i] = hit_id
        copy(mempool.hit_reflectance[i], tri_materials[hit_id][0])
        copy(mempool.hit_emittance[i], tri_materials[hit_id][1])
    elif hit_type == 2: # sphere hit
        i = mempool.depth
        mempool.hit_t[i] = min_t
        axpy(min_t, mempool.ray_d, mempool.ray_o, mempool.hit_p[i])
        mempool.hit_n[i] = (mempool.hit_p[i] - sphere_params[hit_id].c) / sphere_params[hit_id].r
        copy(mempool.hit_tn[i], compute_tangent(mempool.hit_n[i]))
        copy(mempool.hit_bn[i], cross(mempool.hit_n[i], mempool.hit_tn[i]))
        mempool.hit_face_id[i] = hit_id
        copy(mempool.hit_reflectance[i], sphere_materials[hit_id][0])
        copy(mempool.hit_emittance[i], sphere_materials[hit_id][1])

    # two-sided intersection
    i = mempool.depth
    if mempool.valid_hit() and dot(mempool.ray_d, mempool.hit_n[i]) > 0:
        mempool.hit_n[i][0] *= -1
        mempool.hit_n[i][1] *= -1
        mempool.hit_n[i][2] *= -1
