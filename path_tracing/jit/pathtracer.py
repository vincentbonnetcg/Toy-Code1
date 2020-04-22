"""
@author: Vincent Bonnet
@description : basic render routines
"""

import time
import math
import random
import numba
import numpy as np
from . import core as jit_core
from .maths import dot, isclose, triple_product, cross, compute_tangent

# pathtracer settings
BLACK = np.zeros(3)
MAX_DEPTH = 1 # max hit
NUM_SAMPLES = 1 # number of sample per pixel
RANDOM_SEED = 10
INV_PDF = 2.0 * math.pi; # inverse of probability density function
INV_PI = 1.0 / math.pi

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
def update_ray_from_uniform_distribution(mempool):
    i = mempool.depth
    copy(mempool.ray_o, mempool.hit_p[i])
    # Find ray direction from uniform around hemisphere
    # Unit hemisphere from spherical coordinates
    # the unit  hemisphere is at origin and y is the up vector
    # theta [0, 2*PI) and phi [0, PI/2]
    # px = cos(theta)*sin(phi)
    # py = sin(theta)*sin(phi)
    # pz = cos(phi)
    # A uniform distribution (avoid more samples at the pole)
    # theta = 2*PI*rand()
    # phi = acos(rand())  not phi = PI/2*rand() !
    # Optimization
    # cos(phi) = cos(acos(rand())) = rand()
    # sin(phi) = sin(acos(rand())) = sqrt(1 - rand()^2)
    theta = 2*math.pi*random.random()
    cos_phi = random.random()
    sin_phi = math.sqrt(1.0 - cos_phi**2)
    v0 = math.cos(theta)*sin_phi
    v1 = cos_phi
    v2 = math.sin(theta)*sin_phi
    # compute the world sample
    mempool.ray_d[0] = v0*mempool.hit_bn[i][0] + v1*mempool.hit_n[i][0] + v2*mempool.hit_tn[i][0]
    mempool.ray_d[1] = v0*mempool.hit_bn[i][1] + v1*mempool.hit_n[i][1] + v2*mempool.hit_tn[i][1]
    mempool.ray_d[2] = v0*mempool.hit_bn[i][2] + v1*mempool.hit_n[i][2] + v2*mempool.hit_tn[i][2]

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

@numba.njit
def ray_details(details, mempool, skip_face_id = -1):
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
        t = ray_quad(mempool, quad_vertices[i])
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
        t = ray_triangle(mempool, tri_vertices[i])
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
        t = ray_sphere(mempool, c, r)
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

@numba.njit
def recursive_trace(details, mempool):
    if mempool.depth + 1 >= MAX_DEPTH: # can another hit be allocated ?
        return BLACK

    skip_face_id = mempool.hit_face_id[mempool.depth]
    ray_details(details, mempool, skip_face_id)
    if not mempool.valid_hit():
        return BLACK

    # update ray and compute weakening factor
    depth = mempool.depth
    update_ray_from_uniform_distribution(mempool)
    weakening_factor = dot(mempool.ray_d, mempool.hit_n[depth])

    # compute rendering equation
    #BRDF = hit.reflectance / math.pi
    return mempool.hit_emittance[depth] + ((mempool.hit_reflectance[depth] * INV_PI) *
                                           recursive_trace(details, mempool) *
                                           weakening_factor * INV_PDF)

@numba.njit
def first_trace(details, mempool):
    if MAX_DEPTH == 0:
        return mempool.hit_reflectance[0]

    # update ray and compute weakening factor
    mempool.depth = 0
    update_ray_from_uniform_distribution(mempool)
    weakening_factor = dot(mempool.ray_d, mempool.hit_n[0])

    # compute rendering equation
    #BRDF =  hit.reflectance / math.pi
    return mempool.hit_emittance[0] + ((mempool.hit_reflectance[0] * INV_PI) *
                                       recursive_trace(details, mempool) *
                                       weakening_factor * INV_PDF)

@numba.njit
def render(image, camera, details, start_time):
    mempool = jit_core.MemoryPool(NUM_SAMPLES)
    random.seed(RANDOM_SEED)
    for j in range(camera.height):
        for i in range(camera.width):
            # compute first hit to the scene
            camera.get_ray(i, j, mempool)
            ray_details(details, mempool)

            if mempool.valid_hit() == False:
                continue

            for _ in range(NUM_SAMPLES):
                pixel_shade = first_trace(details, mempool)
                image[camera.height-1-j, camera.width-1-i] += pixel_shade

            image[camera.height-1-j, camera.width-1-i] /= NUM_SAMPLES

        with numba.objmode():
            p = (j+1) / camera.height
            print('. completed : %.2f' % (p * 100.0), ' %')
            if time.time() != start_time:
                t = time.time() - start_time
                estimated_time_left = (1.0 - p) / p * t
                print('    estimated time left: %.2f sec' % estimated_time_left)

    print('Total intersections ', mempool.total_intersection)
