"""
@author: Vincent Bonnet
@description : objects to describe a scene
"""

import numpy as np

import geometry
from jit import core as jit_core
from jit import maths as jit_maths

class Material:
    def __init__(self, rgb = [1,1,1]):
        self.d = np.asarray(rgb) / 255.0 # diffuse

class Sphere():
    def __init__(self, center = np.zeros(3), radius = 1.0):
        self.c = center
        self.r = radius
        self.material = Material([140, 255, 191])

    def intersect(self, ray : jit_core.Ray):
        t = jit_maths.ray_sphere(ray.o, ray.d, self.c, self.r)
        hit = jit_core. Hit(t)
        if hit.valid():
            hit.p = ray.o + (ray.d * t)
            hit.n = (hit.p - self.c) / self.r
            hit.diffuse = self.material.d

        return hit

class PolygonMesh():
    def __init__(self):
        self.v, self.t, self.n = geometry.create_test_triangle(-2)
        self.material = Material([232, 232, 128])

    def intersect(self, ray : jit_core.Ray):
        min_t = np.finfo(np.float).max
        hit = jit_core.Hit()
        triangle_vertices = np.zeros((3, 3), dtype=float)
        for ti in range(len(self.t)):
            np.copyto(triangle_vertices[0], self.v[self.t[ti][0]])
            np.copyto(triangle_vertices[1], self.v[self.t[ti][1]])
            np.copyto(triangle_vertices[2], self.v[self.t[ti][2]])
            t = jit_maths.ray_triangle(ray.o, ray.d, triangle_vertices)
            if t > 0.0 and t < min_t:
                hit.t = t
                hit.p = ray.o + (ray.d * t)
                hit.n = self.n[ti]
                hit.diffuse = self.material.d
                min_t = t

        return hit

class AreaLight():
    def __init__(self):
        pass

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []

    def load(self):
        # create sphere
        sphere_center = np.zeros(3)
        np.copyto(sphere_center, [0, -0.1, -2])
        sphere_radius = 0.5
        sphere = Sphere(sphere_center, sphere_radius)
        self.objects.append(sphere)
        # create polygon mesh
        polygon = PolygonMesh()
        self.objects.append(polygon)
        # create lights
        light = AreaLight()
        self.lights.append(light)

    def intersect(self, ray : jit_core.Ray):
        hit = jit_core.Hit()  # intersection
        for obj in self.objects:
            obj_hit = obj.intersect(ray)
            if (obj_hit.valid() and obj_hit.t < hit.t) or not hit.valid():
                hit = obj_hit
        return hit