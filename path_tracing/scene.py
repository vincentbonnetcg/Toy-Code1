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
        hit = jit_core.Hit(t)
        if hit.valid():
            hit.p = ray.o + (ray.d * t)
            hit.n = (hit.p - self.c) / self.r
            hit.diffuse = self.material.d

        return hit

class PolygonSoup():
    def __init__(self):
        self.tv = None # triangle vertices
        self.n = None  # triangle normal
        self.material = Material([232, 232, 128])
        v, ti, n = geometry.create_test_triangle(-2)
        self.set_polygons(v, ti, n)

    def set_polygons(self, vertices, triangle_indices, normals):
        self.tv = np.take(vertices, triangle_indices, axis=0)
        self.n = normals

    def get_triangles(self):
        result = []
        for triVertices in self.tv:
            result.append(triVertices)
        return result

    def get_normals(self):
        result = []
        for triNormal in self.n:
            result.append(triNormal)
        return result

    def intersect(self, ray : jit_core.Ray):
        min_t = np.finfo(np.float).max
        hit = jit_core.Hit()
        for i in range(len(self.tv)):
            t = jit_maths.ray_triangle(ray.o, ray.d, self.tv[i])
            if t > 0.0 and t < min_t:
                hit.t = t
                hit.p = ray.o + (ray.d * t)
                hit.n = self.n[i]
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
        #sphere_center = np.zeros(3)
        #np.copyto(sphere_center, [0, -0.1, -2])
        #sphere_radius = 0.5
        #sphere = Sphere(sphere_center, sphere_radius)
        #self.objects.append(sphere)
        # create polygon soup
        polygon = PolygonSoup()
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

    def details(self):
        # gather sphere, triangles and materials
        spheres = []
        triangles = []
        triangle_normals = []
        triangle_materials = []
        for obj in self.objects:
            if isinstance(obj,Sphere):
                spheres.append(obj)
            elif isinstance(obj, PolygonSoup):
                triangles += obj.get_triangles()
                triangle_normals += obj.get_normals()
                triangle_materials.append(obj.material.d)

        # consolidate spheres in contiguous numpy array
        # TODO

        # consolidate triangles in contiguous numpy array
        np_tri_vertices= np.zeros((len(triangles),3,3))
        np_tri_normals = np.zeros((len(triangles),3))
        np_tri_materials = np.zeros((len(triangles),3))
        for ti in range(len(triangles)):
            np_tri_vertices[ti] = triangles[ti]
            np_tri_normals[ti] = triangle_normals[ti]
            np_tri_materials[ti] = triangle_materials[ti]

        print('num_spheres ' , len(spheres))
        print('num_triangles ' , len(triangles))

        details = (np_tri_vertices, np_tri_normals, np_tri_materials)
        return details
