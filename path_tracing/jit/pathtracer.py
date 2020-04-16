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
def update_ray_from_uniform_distribution(mempool, hit):
    mempool.ray_o = hit.p
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
    mempool.ray_d[0] = v0*hit.bn[0] + v1*hit.n[0] + v2*hit.tn[0]
    mempool.ray_d[1] = v0*hit.bn[1] + v1*hit.n[1] + v2*hit.tn[1]
    mempool.ray_d[2] = v0*hit.bn[2] + v1*hit.n[2] + v2*hit.tn[2]

@numba.njit(inline='always')
def _subtract(a, b, out):
    # squeeze some performance by skipping the generic np.subtract
    out[0] = a[0] - b[0]
    out[1] = a[1] - b[1]
    out[2] = a[2] - b[2]

@numba.njit(inline='always')
def ray_triangle(mempool, tv):
    # Moller-Trumbore intersection algorithm
    _subtract(tv[1], tv[0], mempool.v[0]) # e1
    _subtract(tv[2], tv[0], mempool.v[1]) # e2
    _subtract(mempool.ray_o, tv[0], mempool.v[2]) # ed

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
    _subtract(tv[1], tv[0], mempool.v[0]) # e1
    _subtract(tv[2], tv[0], mempool.v[1]) # e2
    _subtract(mempool.ray_o, tv[0], mempool.v[2]) # ed

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
    min_t = np.finfo(numba.float64).max
    hit = jit_core.Hit()
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
        hit.t = min_t
        hit.p = mempool.ray_o + (mempool.ray_d * min_t)
        hit.n = quad_normals[hit_id]
        hit.tn = quad_tangents[hit_id]
        hit.bn = quad_binormals[hit_id]
        hit.face_id = hit_id
        hit.reflectance = quad_materials[hit_id][0]
        hit.emittance = quad_materials[hit_id][1]
        mempool.num_hit += 1
    elif hit_type == 1: # triangle hit
        hit.t = min_t
        hit.p = mempool.ray_o + (mempool.ray_d * min_t)
        hit.n = tri_normals[hit_id]
        hit.tn = tri_tangents[hit_id]
        hit.bn = tri_binormals[hit_id]
        hit.face_id = hit_id
        hit.reflectance = tri_materials[hit_id][0]
        hit.emittance = tri_materials[hit_id][1]
        mempool.num_hit += 1
    elif hit_type == 2: # sphere hit
        hit.t = min_t
        hit.p = mempool.ray_o + (mempool.ray_d * min_t)
        hit.n = (hit.p - sphere_params[hit_id].c) / sphere_params[hit_id].r
        hit.tn = compute_tangent(hit.n)
        hit.bn = cross(hit.n, hit.tn)
        hit.face_id = hit_id
        hit.reflectance = sphere_materials[hit_id][0]
        hit.emittance = sphere_materials[hit_id][1]
        mempool.num_hit += 1

    # two-sided intersection
    if hit.valid() and dot(mempool.ray_d, hit.n) > 0:
        hit.n[0] *= -1
        hit.n[1] *= -1
        hit.n[2] *= -1

    return hit

@numba.njit
def recursive_trace(details, mempool, count_depth=0, skip_face_id=-1):
    if count_depth >= MAX_DEPTH:
        return BLACK

    hit = ray_details(details, mempool, skip_face_id)
    if not hit.valid():
        return BLACK

    # update ray and compute weakening factor
    update_ray_from_uniform_distribution(mempool, hit)
    weakening_factor = dot(mempool.ray_d, hit.n)

    # compute incoming light
    incoming = recursive_trace(details, mempool, count_depth+1, hit.face_id)

    # compute rendering equation
    #BRDF = hit.reflectance / math.pi
    return hit.emittance + ((hit.reflectance * INV_PI) * incoming * weakening_factor * INV_PDF)

@numba.njit
def first_trace(hit, details, mempool):
    if MAX_DEPTH == 0:
        return hit.reflectance

    # update ray and compute weakening factor
    update_ray_from_uniform_distribution(mempool, hit)
    weakening_factor = dot(mempool.ray_d, hit.n)

    # compute incoming light
    count_depth = 0
    incoming = recursive_trace(details, mempool, count_depth+1, hit.face_id)

    # compute rendering equation
    #BRDF =  hit.reflectance / math.pi
    return hit.emittance + ((hit.reflectance * INV_PI) * incoming * weakening_factor * INV_PDF)

@numba.njit
def render(image, camera, details, start_time):
    mempool = jit_core.MemoryPool()
    random.seed(RANDOM_SEED)
    for j in range(camera.height):
        for i in range(camera.width):
            # compute first hit to the scene
            camera.get_ray(i, j, mempool)
            hit = ray_details(details, mempool)
            if not hit.valid():
                continue

            for _ in range(NUM_SAMPLES):
                mempool.num_hit = 1
                pixel_shade = first_trace(hit, details, mempool)
                image[camera.height-1-j, camera.width-1-i] += pixel_shade

            image[camera.height-1-j, camera.width-1-i] /= NUM_SAMPLES

        with numba.objmode():
            p = (j+1) / camera.height
            #print('. completed : %.2f' % (p * 100.0), ' %')
            if time.time() != start_time:
                t = time.time() - start_time
                estimated_time_left = (1.0 - p) / p * t
                #print('    estimated time left: %.2f sec' % estimated_time_left)
    
    print('Total intersection ', mempool.total_intersection)