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
    def __init__(self, material = [1,1,1], mtype=0):
        self.material = np.asarray(material)
        self.materialtype = mtype # 0 reflectance, 1 : emittance

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

    def load_cornell_box(self):
        # From http://www.graphics.cornell.edu/online/box/data.html
        quad_v = [] # quad vertices
        quad_m = [] # quad material
        white = [1,1,1]
        red = [0.57,0.025,0.025]
        green = [0.025,0.236,0.025]
        blue = [0,0,1]
        black = [0,0,0]
        light_intensity = 10.0
        light_colour = [1*light_intensity,0.73*light_intensity,0.4*light_intensity]
        # floor
        quad_v.append([[552.8,0,0],[0,0,0],[0,0,559.2],[549.6,0,559.2]])
        quad_m.append([white, 0])
        # left wall
        quad_v.append([[552.8,0,0],[549.6,0,559.2],[556,548.8,559.2],[556,548.8,0]])
        quad_m.append([red, 0])
        # right wall
        quad_v.append([[0,0,559.2],[0,0,0],[0,548.8,0],[0,548.8,559.2]])
        quad_m.append([green, 0])
        # back wall
        quad_v.append([[549.6,0,559.2],[0,0,559.2],[0,548.8,559.2],[556,548.8,559.2]])
        quad_m.append([white, 0])
        # ceiling (large light)
        quad_v.append([[556,548.8,0],[556,548.8,559.2],[0,548.8,559.2],[0,548.8,0]])
        quad_m.append([white, 0])
        # short block
        quad_v.append([[130,165,65],[82,165,225],[240,165,272],[290,165,114]])
        quad_m.append([white, 0])
        quad_v.append([[290,0,114],[290,165,114],[240,165,272],[240,0,272]])
        quad_m.append([white, 0])
        quad_v.append([[130,0,65],[130,165,65],[290,165,114],[290,0,114]])
        quad_m.append([white, 0])
        quad_v.append([[82,0,225],[82,165,225],[130,165,65],[130,0,65]])
        quad_m.append([white, 0])
        quad_v.append([[240,0,272],[240,165,272],[82,165,225],[82,0,225]])
        quad_m.append([white, 0])
        # tall block
        quad_v.append([[423,330,247],[265,330,296],[314,330,456],[472,330,406]])
        quad_m.append([white, 0])
        quad_v.append([[423,0,247],[423,330,247],[472,330,406],[472,0,406]])
        quad_m.append([white, 0])
        quad_v.append([[472,0,406],[472,330,406],[314,330,456],[314,0,456]])
        quad_m.append([white, 0])
        quad_v.append([[314,0,456],[314,330,456],[265,330,296],[265,0,296]])
        quad_m.append([white, 0])
        quad_v.append([[265,0,296],[265,330,296],[423,330,247],[423,0,247]])
        quad_m.append([white, 0])
        # small light
        # added an offset from the cornell box from 548.8 to 548
        quad_v.append([[343,548.79,227],[343,548.79,332],[213,548.79,332],[213,548.79,227]])
        quad_m.append([light_colour, 1])

        # add quads
        for i in range(len(quad_v)):
            tv, n = geometry.create_tri_quad(quad_v[i])
            material = Material(quad_m[i][0], quad_m[i][1])
            self.objects.append(TriangleSoup(tv, n, material))
        # set camera
        np.copyto(self.camera.origin, [278, 273, -800])
        self.camera.dir_z = 1.0
        focal_length = 35 # in mm
        sensor_size = 25 # in mm (sensor width and height)
        self.camera.fovx = math.atan(sensor_size*0.5/focal_length) * 2

    def tri_details(self):
        # gather sphere, triangles and materials
        num_triangles = 0
        triangles = []
        for obj in self.objects:
            if isinstance(obj, TriangleSoup):
                num_triangles += obj.num_triangles()
                triangles.append(obj)

        # numpy dtype to store structure of array
        dtype_dict = {}
        dtype_dict['names'] = ['tri_vertices', 'tri_normals', 'tri_tangents',
                               'tri_binormals', 'tri_materials', 'tri_materialtype']
        dtype_dict['formats'] = []
        dtype_dict['formats'].append((np.float32, (num_triangles,3,3)))
        dtype_dict['formats'].append((np.float32, (num_triangles,3)))
        dtype_dict['formats'].append((np.float32, (num_triangles,3)))
        dtype_dict['formats'].append((np.float32, (num_triangles,3)))
        dtype_dict['formats'].append((np.float32, (num_triangles,3)))
        dtype_dict['formats'].append((np.int32, num_triangles))
        tri_data = np.zeros(1, dtype=np.dtype(dtype_dict, align=True))

        # consolidate triangles in contiguous numpy array
        index = 0
        for tri in triangles:
            data = tri_data[0]
            for i in range(len(tri.tv)):
                data['tri_vertices'][index] = tri.tv[i]
                data['tri_normals'][index] = tri.n[i]
                data['tri_materialtype'][index] = tri.material.materialtype
                data['tri_materials'][index] = tri.material.material
                index += 1

        jit_math.compute_tangents_binormals(tri_data[0]['tri_normals'],
                                            tri_data[0]['tri_tangents'],
                                            tri_data[0]['tri_binormals'])

        print('num_triangles ' , num_triangles)

        return tri_data
