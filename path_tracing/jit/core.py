"""
@author: Vincent Bonnet
@description : core objects not used to describe a scene
"""

import math
import numpy as np
import numba
from jit.maths import normalize

# A per-thread fixed memory pool to prevent memory allocation  and contains
# . pre-allocated arrays
# . pre-allocated ray (origin, direction)
# . pre_allocated hits
@numba.jitclass([('v', numba.float64[:,:]),
                 ('ray_o', numba.float64[:]),
                 ('ray_d', numba.float64[:]),
                 ('depth', numba.int32),
                 ('total_intersection', numba.int32)])
class MemoryPool:
    def __init__(self):
        self.v = np.empty((3,3)) # pool of vectors
        self.ray_o = np.empty(3) # used for ray origin
        self.ray_d = np.empty(3) # used for ray direction
        self.depth = 0         # depth counter
        self.total_intersection = 0    # total number ray vs element intersection

@numba.jitclass([('t', numba.float64), # ray distance as double
                 ('p', numba.float64[:]), # hit positon as np.empty(3)
                 ('n', numba.float64[:]), # hit normal as np.empty(3)
                 ('tn', numba.float64[:]), # hit tangent as np.empty(3)
                 ('bn', numba.float64[:]), # hit binormal as np.empty(3)
                 ('face_id', numba.int32), # face id
                 ('reflectance', numba.float64[:]), # reflectance as np.empty(3)
                 ('emittance', numba.float64[:])]) # emittance as np.empty(3)
class Hit:
    def __init__(self):
        self.t = -1.0 # ray distance
        self.face_id = -1

    def valid(self):
        if self.t >= 0.0:
            return True
        return False

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

    def get_ray(self, i : int, j : int, mempool):
        x = (2 * i - (self.width-1)) / (self.width-1) * self.tan_fovx
        y = (2 * j - (self.height-1)) / (self.height-1) * self.tan_fovy
        mempool.ray_o[0] = self.origin[0]
        mempool.ray_o[1] = self.origin[1]
        mempool.ray_o[2] = self.origin[2]
        mempool.ray_d[0] = x
        mempool.ray_d[1] = y
        mempool.ray_d[2] = self.dir_z
        normalize(mempool.ray_d)
