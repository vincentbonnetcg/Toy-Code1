"""
@author: Vincent Bonnet
@description : Pathtracer with Python+Numba
"""

import math
import IPython.display

import numpy as np
import io
import PIL

import common
from jit import core as jit_core
from scene_details import Scene

NUM_SAMPLES = 1 # number of sample per pixel

def shade(ray : jit_core.Ray, hit : jit_core.Hit):
    if hit.valid():
        dot =  math.fabs(np.dot(ray.d, hit.n))
        return hit.diffuse * dot
    # background colour
    return np.asarray([10,10,10])/255.0

def trace(ray : jit_core.Ray, scene : Scene):
    hit = scene.intersect(ray)
    return shade(ray, hit)

@common.timeit
def render(scene : Scene, camera : jit_core.Camera):
    image = np.zeros((camera.height, camera.width, 3))
    for i in range(camera.width):
        for j in range(camera.height):
            for _ in range(NUM_SAMPLES):
                ray = camera.ray(i, j)
                image[camera.height-1-j, i] = trace(ray, scene)
    return image

@common.timeit
def show(image):
    buffer = io.BytesIO()
    PIL.Image.fromarray(np.uint8(image*255)).save(buffer, 'png')
    IPython.display.display(IPython.display.Image(data=buffer.getvalue()))

def main():
    scene = Scene()
    scene.load()
    camera = jit_core.Camera(320, 240)
    image = render(scene, camera)
    show(image)

if __name__ == '__main__':
    main()

