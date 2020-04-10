"""
@author: Vincent Bonnet
@description : basic render routines
"""

import math
import numba
import numpy as np
from . import core as jit_core
from .maths import dot, isclose, triple_product

@numba.njit(inline='always')
def _subtract(a, b, out):
    # squeeze some performance by skipping the generic np.subtract
    out[0] = a[0] - b[0]
    out[1] = a[1] - b[1]
    out[2] = a[2] - b[2]

@numba.njit(inline='always')
def ray_triangle(ray_o, ray_d, tv, edges):
    # Moller-Trumbore intersection algorithm
    _subtract(tv[1], tv[0], edges[0]) # e1
    _subtract(tv[2], tv[0], edges[1]) # e2
    _subtract(ray_o, tv[0], edges[2]) # ed

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
    detA = -triple_product(ray_d, edges[0], edges[1])
    if isclose(detA, 0.0):
        # ray is parallel to the triangle
        return -1.0

    invDetA = 1.0 / detA

    u = -triple_product(ray_d, edges[2], edges[1]) * invDetA
    if (u < 0.0 or u > 1.0):
        return -1.0

    v = -triple_product(ray_d, edges[0], edges[2]) * invDetA
    if (v < 0.0 or u + v > 1.0):
        return -1.0

    return triple_product(edges[2], edges[0], edges[1]) * invDetA # t

@numba.njit(inline='always')
def ray_quad(ray_o, ray_d, tv, edges):
    # Moller-Trumbore intersection algorithm
    # same than ray_triangle but different condition on v
    _subtract(tv[1], tv[0], edges[0]) # e1
    _subtract(tv[2], tv[0], edges[1]) # e2
    _subtract(ray_o, tv[0], edges[2]) # ed

    detA = -triple_product(ray_d, edges[0], edges[1])
    if isclose(detA, 0.0):
        # ray is parallel to the triangle
        return -1.0

    invDetA = 1.0 / detA

    u = -triple_product(ray_d, edges[2], edges[1]) * invDetA
    if (u < 0.0 or u > 1.0):
        return -1.0

    v = -triple_product(ray_d, edges[0], edges[2]) * invDetA
    if (v < 0.0 or v > 1.0):
        return -1.0

    return triple_product(edges[2], edges[0], edges[1]) * invDetA # t

@numba.njit(inline='always')
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
def ray_details(ray, details, skip_face_id = -1):
    min_t = np.finfo(numba.float64).max
    hit = jit_core.Hit()
    quad_vertices = details[0]
    quad_normals = details[1]
    quad_materials = details[2]
    tri_vertices = details[3]
    tri_normals = details[4]
    tri_materials = details[5]
    sphere_params = details[6]
    sphere_materials = details[7]
    hit_type = -1
    hit_id = -1
    # edges is a preallocated cache to prevent memory allocation
    # during the critical intersection part (twice faster)
    edges = np.empty((3, 3))
    # intersection test with triangles
    num_quads = len(quad_vertices)
    for i in range(num_quads):
        if i == skip_face_id:
            continue
        t = ray_quad(ray.o, ray.d, quad_vertices[i], edges)
        if t > 0.0 and t < min_t:
            min_t = t
            hit_type = 0
            hit_id = i

    # intersection test with triangles
    num_triangles = len(tri_vertices)
    for i in range(num_triangles):
        if i == skip_face_id:
            continue
        t = ray_triangle(ray.o, ray.d, tri_vertices[i], edges)
        if t > 0.0 and t < min_t:
            min_t = t
            hit_type = 1
            hit_id = i

    # intersection test with spheres
    num_spheres = len(sphere_params)
    for i in range(num_spheres):
        c = sphere_params[i].c
        r = sphere_params[i].r
        t = ray_sphere(ray.o, ray.d, c, r)
        if t > 0.0 and t < min_t:
            min_t = t
            hit_type = 2
            hit_id = i

    if hit_type == 0: # quad hit
        hit.t = min_t
        hit.p = ray.o + (ray.d * min_t)
        hit.n = quad_normals[hit_id]
        hit.face_id = hit_id
        hit.reflectance = quad_materials[hit_id][0]
        hit.emittance = quad_materials[hit_id][1]
    elif hit_type == 1: # triangle hit
        hit.t = min_t
        hit.p = ray.o + (ray.d * min_t)
        hit.n = tri_normals[hit_id]
        hit.face_id = hit_id
        hit.reflectance = tri_materials[hit_id][0]
        hit.emittance = tri_materials[hit_id][1]
    elif hit_type == 2: # sphere hit
        hit.t = min_t
        hit.p = ray.o + (ray.d * min_t)
        hit.n = (hit.p - sphere_params[hit_id].c) / sphere_params[hit_id].r
        hit.face_id = hit_id
        hit.reflectance = sphere_materials[hit_id][0]
        hit.emittance = sphere_materials[hit_id][1]

    return hit


@numba.njit
def shade(ray : jit_core.Ray, hit : jit_core.Hit):
    if hit.valid():
        return hit.reflectance * math.fabs(dot(ray.d, hit.n))
    # background colour
    return np.asarray([10,10,10])/255.0

@numba.njit
def trace(ray : jit_core.Ray, details):
    hit = ray_details(ray, details)
    return shade(ray, hit)

@numba.njit
def render(image, camera, details, num_samples):
    ray = jit_core.Ray()
    for j in range(camera.height):
        for i in range(camera.width):
            for _ in range(num_samples):
                camera.get_ray(i, j, ray)
                image[camera.height-1-j, camera.width-1-i] = trace(ray, details)
