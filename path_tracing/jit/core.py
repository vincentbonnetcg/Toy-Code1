"""
@author: Vincent Bonnet
@description : core objects not used to describe a scene
"""

import math
import numpy as np
import numba

@numba.jitclass([('t', numba.float64), # ray distance as double
                 ('p', numba.float64[:]), # hit positon as np.zeros(3)
                 ('n', numba.float64[:]), # hit normal as np.zeros(3)
                 ('diffuse', numba.float64[:])]) # diffuse material as np.zeros(3)
class Hit:
    def __init__(self):
        self.t = -1.0 # ray distance

    def valid(self):
        if self.t >= 0.0:
            return True
        return False

@numba.jitclass([('o', numba.float64[:]),
                 ('d', numba.float64[:])])
class Ray:
    def __init__(self):
        pass

@numba.jitclass([('origin', numba.float64[:]),
                 ('width', numba.int32),
                 ('height', numba.int32),
                 ('fovx', numba.float64),
                 ('fovy', numba.float64),
                 ('tan_fovx', numba.float64),
                 ('tan_fovy', numba.float64),
                 ('dir_z', numba.float64)])
class Camera:
    def __init__(self, width : int, height : int):
        self.origin = np.zeros(3)
        self.fovx = np.pi / 2
        self.dir_z = -1.0
        self.set_resolution(width, height)

    def set_resolution(self, width : int, height : int):
        self.width = width
        self.height = height
        self.fovy = float(self.height) / float(self.width) * self.fovx
        self.tan_fovx = math.tan(self.fovx*0.5)
        self.tan_fovy = math.tan(self.fovy*0.5)

    def ray(self, i : int, j : int):
        x = (2 * i - (self.width-1)) / (self.width-1) * self.tan_fovx
        y = (2 * j - (self.height-1)) / (self.height-1) * self.tan_fovy
        ray = Ray()
        ray.o = self.origin
        ray.d = np.empty(3)
        ray.d[0] = x
        ray.d[1] = y
        ray.d[2] = self.dir_z
        invnorm = 1.0 / math.sqrt(ray.d[0]*ray.d[0]+
                                  ray.d[1]*ray.d[1]+
                                  ray.d[2]*ray.d[2])
        ray.d[0] *= invnorm
        ray.d[1] *= invnorm
        ray.d[2] *= invnorm
        return ray

