"""
@author: Vincent Bonnet
@description : objects to describe a scene
"""

import math
import numpy as np
import geometry
from jit import core as jit_core

class Material:
    def __init__(self, rgb = [1,1,1]):
        self.d = np.asarray(rgb) # diffuse

class Sphere():
    def __init__(self, center = np.zeros(3), radius = 1.0):
        self.c = center
        self.r = radius
        self.material = Material([0.549, 1.0, 0.749])

class PolygonSoup():
    def __init__(self, triangle_vertices, normals, diffuse=[0.91, 0.91, 0.5]):
        self.tv = triangle_vertices
        self.n = normals  # triangle normal
        self.material = Material(diffuse)

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

    def get_materials(self):
        result = []
        for _ in range(len(self.n)):
            result.append(self.material.d)
        return result

class AreaLight():
    def __init__(self):
        pass

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.camera = jit_core.Camera(320, 240)

    def load(self):
        # create sphere
        sphere_center = np.zeros(3)
        np.copyto(sphere_center, [0, -0.1, -2])
        sphere_radius = 0.5
        sphere = Sphere(sphere_center, sphere_radius)
        self.objects.append(sphere)
        # create polygon soup
        tv, n = geometry.create_test_triangle(-2)
        polygon = PolygonSoup(tv, n)
        self.objects.append(polygon)
        # create lights
        light = AreaLight()
        self.lights.append(light)

    def load_cornell_box(self):
        # From http://www.graphics.cornell.edu/online/box/data.html
        quad_v = [] # quad vertices
        quad_m = [] # quad material
        # floor
        quad_v.append([[552.8,0,0],[0,0,0],[0,0,559.2],[549.6,0,559.2]])
        quad_m.append([1,1,1])
        # left wall
        quad_v.append([[552.8,0,0],[549.6,0,559.2],[556,548.8,559.2],[556,548.8,0]])
        quad_m.append([1,0,0])
        # right wall
        quad_v.append([[0,0,559.2],[0,0,0],[0,548.8,0],[0,548.8,559.2]])
        quad_m.append([0,1,0])
        # back wall
        quad_v.append([[549.6,0,559.2],[0,0,559.2],[0,548.8,559.2],[556,548.8,559.2]])
        quad_m.append([1,1,1])
        # ceiling
        quad_v.append([[556,548.8,0],[556,548.8,559.2],[0,548.8,559.2],[0,548.8,0]])
        quad_m.append([1,1,1])
        # short block
        quad_v.append([[130,165,65],[82,165,225],[240,165,272],[290,165,114]])
        quad_m.append([1,1,1])
        quad_v.append([[290,0,114],[290,165,114],[240,165,272],[240,0,272]])
        quad_m.append([1,1,1])
        quad_v.append([[130,0,65],[130,165,65],[290,165,114],[290,0,114]])
        quad_m.append([1,1,1])
        quad_v.append([[82,0,225],[82,165,225],[130,165,65],[130,0,65]])
        quad_m.append([1,1,1])
        quad_v.append([[240,0,272],[240,165,272],[82,165,225],[82,0,225]])
        quad_m.append([1,1,1])
        # tall block
        quad_v.append([[423,330,247],[265,330,296],[314,330,456],[472,330,406]])
        quad_m.append([1,1,1])
        quad_v.append([[423,0,247],[423,330,247],[472,330,406],[472,0,406]])
        quad_m.append([1,1,1])
        quad_v.append([[472,0,406],[472,330,406],[314,330,456],[314,0,456]])
        quad_m.append([1,1,1])
        quad_v.append([[314,0,456],[314,330,456],[265,330,296],[265,0,296]])
        quad_m.append([1,1,1])
        quad_v.append([[265,0,296],[265,330,296],[423,330,247],[423,0,247]])
        quad_m.append([1,1,1])
        # add quads
        for i in range(len(quad_v)):
            tv, n = geometry.create_quad(quad_v[i])
            self.objects.append(PolygonSoup(tv, n, quad_m[i]))
        # create lights
        light = AreaLight()
        self.lights.append(light)
        # set camera
        np.copyto(self.camera.origin, [278, 273, -800])
        self.camera.dir_z = 1.0
        focal_length = 35 # in mm
        sensor_size = 25 # in mm (sensor width and height)
        self.camera.fovx = math.atan(sensor_size*0.5/focal_length) * 2

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
                triangle_materials += obj.get_materials()

        # consolidate spheres in contiguous numpy array
        sphere_dtype = np.dtype([('c', np.float64, (3,)), ('r', np.float64)])
        np_sph_params = np.zeros(len(spheres), dtype =sphere_dtype)
        np_sph_materials = np.zeros((len(spheres),3))
        for si in range(len(spheres)):
            np_sph_params[si]['c'] = spheres[si].c
            np_sph_params[si]['r'] = spheres[si].r
            np_sph_materials[si] = spheres[si].material.d

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

        details = (np_tri_vertices, np_tri_normals, np_tri_materials, np_sph_params, np_sph_materials)
        return details
