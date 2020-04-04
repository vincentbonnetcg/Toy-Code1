"""
@author: Vincent Bonnet
@description : core objects not used to describe a scene
"""

import numpy as np
import numba

@numba.jitclass([('t', numba.float64),
                 ('p', numba.float64[:]),
                 ('n', numba.float64[:]),
                 ('diffuse', numba.float64[:])])
class Hit:
    def __init__(self, t = -1.0):
        self.t = t # ray distance
        self.p = np.zeros(3) # hit positon
        self.n = np.zeros(3) # hit normal
        self.diffuse = np.zeros(3) # diffuse material

    def valid(self):
        if self.t >= 0.0:
            return True
        return False

@numba.jitclass([('o', numba.float64[:]),
                 ('d', numba.float64[:])])
class Ray:
    def __init__(self, origin, direction):
        self.o = np.zeros(3)
        self.d = np.zeros(3)
        self.o = origin
        self.d = direction / np.linalg.norm(direction)


@numba.jitclass([('origin', numba.float64[:]),
                 ('width', numba.int32),
                 ('height', numba.int32),
                 ('fovx', numba.float64),
                 ('fovy', numba.float64)])
class Camera:
    def __init__(self, width : int, height : int):
        self.origin = np.zeros(3)
        self.width = width
        self.height = height
        self.fovx = np.pi / 2
        self.fovy = float(self.height) / float(self.width) * self.fovx

    def ray(self, i : int, j : int):
        x = (2 * i - (self.width-1)) / (self.width-1) * np.tan(self.fovx*0.5)
        y = (2 * j - (self.height-1)) / (self.height-1) * np.tan(self.fovy*0.5)
        direction = np.zeros(3)
        direction[0] = x
        direction[1] = y
        direction[2] = -1
        return Ray(self.origin, direction)



