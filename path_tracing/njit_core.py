"""
@author: Vincent Bonnet
@description : core objects not used to describe a scene
"""
import numpy as np

class Ray:
    def __init__(self, orgin, direction):
        self.o = orgin
        self.d = direction / np.linalg.norm(direction)

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


