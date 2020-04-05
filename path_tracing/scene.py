"""
@author: Vincent Bonnet
@description : objects to describe a scene
"""

import numpy as np
import geometry

class Material:
    def __init__(self, rgb = [1,1,1]):
        self.d = np.asarray(rgb) / 255.0 # diffuse

class Sphere():
    def __init__(self, center = np.zeros(3), radius = 1.0):
        self.c = center
        self.r = radius
        self.material = Material([140, 255, 191])

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
        # create polygon soup
        polygon = PolygonSoup()
        self.objects.append(polygon)
        # create lights
        light = AreaLight()
        self.lights.append(light)

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
