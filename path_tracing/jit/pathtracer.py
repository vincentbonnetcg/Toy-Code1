"""
@author: Vincent Bonnet
@description : basic render routines
"""

import math
import random
import numba
import numpy as np
from . import core as jit_core
from .maths import dot, isclose, triple_product, cross

# pathtracer settings
BLACK = np.zeros(3)
MAX_DEPTH = 1 # max hit
NUM_SAMPLES = 1 # number of sample per pixel
RANDOM_SEED = 10
INV_PDF =  2.0*math.pi; # inverse of probability density function

@numba.njit(inline='always')
def get_uniform_sample_around_normal(n):
    # Unit hemisphere from spherical coordinates
    # the unit  hemisphere is at origin and y is the up vector
    # theta [0, 2*PI) and phi [0, PI/2]
    # px = cos(theta)*sin(phi)
    # py = sin(theta)*sin(phi)
    # pz = cos(phi)
    # A uniform distribution (avoid more samples at the pole)
    # theta = 2*PI*rand()
    # phi = acos(rand())  not phi = PI/2*rand() !
    theta = 2*math.pi*random.random()
    phi = math.acos(random.random())
    v = [math.cos(theta)*math.sin(phi),
         math.cos(phi),
         math.sin(theta)*math.sin(phi)]
    # compute local coordinate system
    nt = [0.,0.,0.]
    if abs(n[0]) > abs(n[1]):
        ntdot = n[0]**2+n[2]**2
        nt[0] = n[2]/ntdot
        nt[2] = -n[0]/ntdot
    else:
        ntdot = n[1]**2+n[2]**2
        nt[1] = -n[2]/ntdot
        nt[2] = n[1]/ntdot
    nb = cross(n, nt)
    # compute the world sample
    wv = [0., 0., 0.]
    wv[0] = v[0]*nb[0] + v[1]*n[0] + v[2]*nt[0]
    wv[1] = v[0]*nb[1] + v[1]*n[1] + v[2]*nt[1]
    wv[2] = v[0]*nb[2] + v[1]*n[2] + v[2]*nt[2]

    return np.asarray(wv)

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
def trace(ray : jit_core.Ray, details, count_depth=0, skip_face_id = -1):
    if count_depth >= MAX_DEPTH:
        return BLACK

    hit = ray_details(ray, details, skip_face_id)
    if not hit.valid():
        return BLACK

    # compute new ray from hit
    new_ray = jit_core.Ray()
    new_ray.o = hit.p
    new_ray.d = get_uniform_sample_around_normal(hit.n)

    # compute incoming light
    incoming = trace(new_ray, details, count_depth+1, hit.face_id)

    # compute rendering equation
    brdf =  hit.reflectance / math.pi
    return hit.emittance + (brdf * incoming * dot(new_ray.d, hit.n) * INV_PDF)

@numba.njit
def render(image, camera, details):
    random.seed(RANDOM_SEED)
    ray = jit_core.Ray()
    for j in range(camera.height):
        for i in range(camera.width):
            for _ in range(NUM_SAMPLES):
                camera.get_ray(i, j, ray)
                pixel_shade = trace(ray, details)
                image[camera.height-1-j, camera.width-1-i] += pixel_shade

            image[camera.height-1-j, camera.width-1-i] /= NUM_SAMPLES
