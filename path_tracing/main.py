"""
@author: Vincent Bonnet
@description : Pathtracer with Python+Numba
"""

import math
import njit_utils
import numpy as np
import numba
import matplotlib
import matplotlib.pyplot as plt

NUM_SAMPLES = 1 # number of sample per pixel

class Ray:
    def __init__(self, orgin, direction):
        self.o = orgin
        self.d = direction / np.linalg.norm(direction)

class Hit:
    def __init__(self, t = -1.0, p = np.zeros(3), n = np.zeros(3)):
        self.t = t # ray distance
        self.p = p # hit positon
        self.n = n # hit normal

    def valid(self):
        if self.t >= 0.0:
            return True
        return False

class Sphere():
    def __init__(self, center = np.zeros(3), radius = 1.0):
        self.c = center
        self.r = radius

    def intersect(self, ray : Ray):
        t = njit_utils.sphere_intersect(ray.o, ray.d, self.c, self.r)
        hit = Hit(t)
        if hit.valid():
            hit.p = ray.o + (ray.d * t)
            hit.n = (hit.p - self.c) / self.r

        return hit

class AreaLight():
    def __init__(self):
        pass

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

class Scene:
    def __init__(self):
        self.objects = []
        self.lights = []

    def load(self):
        # create objects
        sphere_center = np.zeros(3)
        sphere_center[1] = -0.6
        sphere_center[2] = -2
        sphere_radius = 0.5
        sphere = Sphere(sphere_center, sphere_radius)
        self.objects.append(sphere)
        # create lights
        light = AreaLight()
        self.lights.append(light)

    def intersect(self, ray : Ray):
        hit = Hit()  # intersection
        for obj in self.objects:
            obj_hit = obj.intersect(ray)
            if (obj_hit.valid() and obj_hit.t < hit.t) or not hit.valid():
                hit = obj_hit
        return hit

def shade(ray : Ray, hit : Hit):
    if hit.valid():
        dot =  math.fabs(np.dot(ray.d, hit.n))
        return np.asarray([0.9,0.85,1]) * dot
    return np.asarray([0,0,0])

def trace(ray : Ray, scene : Scene):
    hit = scene.intersect(ray)
    return shade(ray, hit)

def render(scene : Scene, camera : Camera):
    image = np.zeros((camera.height, camera.width, 3))
    for i in range(camera.width):
        for j in range(camera.height):
            for _ in range(NUM_SAMPLES):
                ray = camera.ray(i, j)
                image[camera.height-1-j, i] = trace(ray, scene)

    plt.imshow(image, aspect='auto')
    plt.show()


def main():
    scene = Scene()
    scene.load()
    camera = Camera(320, 240)
    render(scene, camera)

if __name__ == '__main__':
    main()
