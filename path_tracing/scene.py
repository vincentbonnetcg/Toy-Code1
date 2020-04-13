"""
@author: Vincent Bonnet
@description : objects to describe a scene
"""

import math
import numpy as np
import geometry
from jit import core as jit_core
from jit import maths as jit_math

class Material:
    def __init__(self, reflectance = [1,1,1], emittance = [0,0,0]):
        self.reflectance = np.asarray(reflectance)
        self.emittance = np.asarray(emittance)

class Sphere():
    def __init__(self, center, radius, material):
        self.c = np.asarray(center)
        self.r = radius
        self.material = material

class TriangleSoup():
    def __init__(self, triangle_vertices, normals, material):
        self.tv = triangle_vertices
        self.n = normals  # triangle normals
        self.t = None # triangle tangents
        self.b = None # triangle binormals
        self.material = material

    def num_triangles(self):
        return len(self.n)

class QuadSoup():
    def __init__(self, quad_vertices, normals, material):
        self.qv = quad_vertices
        self.n = normals  # quad normals
        self.t = None # quad tangents
        self.b = None # quad binormals
        self.material = material

    def num_quads(self):
        return len(self.n)

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []
        self.camera = jit_core.Camera(320, 240)

    def load_simple_scene(self):
        # create sphere
        material = Material([0.549, 1.0, 0.749])
        sphere = Sphere(center=[0,-0.1,-2], radius=0.5, material=material)
        self.objects.append(sphere)
        # create polygon soup
        material = Material([0.91, 0.91, 0.5])
        tv, n = geometry.create_test_triangle(-2)
        polygon = TriangleSoup(tv, n, material)
        self.objects.append(polygon)

    def load_cornell_box(self):
        # From http://www.graphics.cornell.edu/online/box/data.html
        quad_v = [] # quad vertices
        quad_m = [] # quad material
        white = [1,1,1]
        red = [1,0,0]
        green = [0,1,0]
        blue = [0,0,1]
        black = [0,0,0]
        # floor
        quad_v.append([[552.8,0,0],[0,0,0],[0,0,559.2],[549.6,0,559.2]])
        quad_m.append([white, black])
        # left wall
        quad_v.append([[552.8,0,0],[549.6,0,559.2],[556,548.8,559.2],[556,548.8,0]])
        quad_m.append([red, black])
        # right wall
        quad_v.append([[0,0,559.2],[0,0,0],[0,548.8,0],[0,548.8,559.2]])
        quad_m.append([green, black])
        # back wall
        quad_v.append([[549.6,0,559.2],[0,0,559.2],[0,548.8,559.2],[556,548.8,559.2]])
        quad_m.append([white, black])
        # ceiling (large light)
        quad_v.append([[556,548.8,0],[556,548.8,559.2],[0,548.8,559.2],[0,548.8,0]])
        quad_m.append([black, white])
        # short block
        quad_v.append([[130,165,65],[82,165,225],[240,165,272],[290,165,114]])
        quad_m.append([white, black])
        quad_v.append([[290,0,114],[290,165,114],[240,165,272],[240,0,272]])
        quad_m.append([white, black])
        quad_v.append([[130,0,65],[130,165,65],[290,165,114],[290,0,114]])
        quad_m.append([white, black])
        quad_v.append([[82,0,225],[82,165,225],[130,165,65],[130,0,65]])
        quad_m.append([white, black])
        quad_v.append([[240,0,272],[240,165,272],[82,165,225],[82,0,225]])
        quad_m.append([white, black])
        # tall block
        quad_v.append([[423,330,247],[265,330,296],[314,330,456],[472,330,406]])
        quad_m.append([white, black])
        quad_v.append([[423,0,247],[423,330,247],[472,330,406],[472,0,406]])
        quad_m.append([white, black])
        quad_v.append([[472,0,406],[472,330,406],[314,330,456],[314,0,456]])
        quad_m.append([white, black])
        quad_v.append([[314,0,456],[314,330,456],[265,330,296],[265,0,296]])
        quad_m.append([white, black])
        quad_v.append([[265,0,296],[265,330,296],[423,330,247],[423,0,247]])
        quad_m.append([white, black])
        # small light
        #quad_v.append([[343,548.79,227],[343,548.79,332],[213,548.79,332],[213,548.79,227]])
        #quad_m.append([black, white])

        # add quads
        for i in range(len(quad_v)):
            # use quad : twice faster for this scene
            #tv, n = geometry.create_tri_quad(quad_v[i])
            #self.objects.append(TriangleSoup(tv, n, quad_m[i]))
            tv, n = geometry.create_quad(quad_v[i])
            material = Material(quad_m[i][0], quad_m[i][1])
            self.objects.append(QuadSoup(tv, n, material))
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
        quads = []
        for obj in self.objects:
            if isinstance(obj,Sphere):
                spheres.append(obj)
            elif isinstance(obj, TriangleSoup):
                triangles.append(obj)
            elif isinstance(obj, QuadSoup):
                quads.append(obj)

        # consolidate spheres in contiguous numpy array
        num_spheres = len(spheres)
        sphere_dtype = np.dtype([('c', np.float64, (3,)), ('r', np.float64)])
        sph_params = np.zeros(num_spheres, dtype =sphere_dtype)
        sph_materials = np.zeros((num_spheres,2,3)) # store reflectance and emittance
        for i, sph in enumerate(spheres):
            sph_params[i]['c'] = sph.c
            sph_params[i]['r'] = sph.r
            sph_materials[i][0] = sph.material.reflectance
            sph_materials[i][1] = sph.material.emittance

        # consolidate triangles in contiguous numpy array
        num_triangles = len(triangles)
        tri_vertices= np.zeros((num_triangles,3,3))
        tri_normals = np.zeros((num_triangles,3))
        tri_tangents = np.zeros((num_triangles,3))
        tri_binormals = np.zeros((num_triangles,3))
        tri_materials = np.zeros((num_triangles,2,3))
        for i, tri in enumerate(triangles):
            tri_vertices[i] = tri.tv
            tri_normals[i] = tri.n
            tri_materials[i][0] = tri.material.reflectance
            tri_materials[i][1] = tri.material.emittance
        jit_math.compute_tangents_binormals(tri_normals, tri_tangents, tri_binormals)

        # consolidate triangles in contiguous numpy array
        num_quads = len(quads)
        quad_vertices= np.zeros((num_quads,3,3))
        quad_normals = np.zeros((num_quads,3))
        quad_tangents = np.zeros((num_quads,3))
        quad_binormals = np.zeros((num_quads,3))
        quad_materials = np.zeros((num_quads,2,3))
        for i, quad in enumerate(quads):
            quad_vertices[i] = quad.qv
            quad_normals[i] = quad.n
            quad_materials[i][0] = quad.material.reflectance
            quad_materials[i][1] = quad.material.emittance
        jit_math.compute_tangents_binormals(quad_normals, quad_tangents, quad_binormals)

        print('num_spheres ' , num_spheres)
        print('num_triangles ' , num_triangles)
        print('num_quads ' , num_quads)

        details = (quad_vertices, quad_normals, quad_tangents, quad_binormals, quad_materials,
                    tri_vertices, tri_normals, tri_tangents, tri_binormals, tri_materials,
                    sph_params, sph_materials)
        return details
