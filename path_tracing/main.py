"""
@author: Vincent Bonnet
@description : Pathtracer with Python+Numba
"""

import numba
import math
import IPython.display

import numpy as np
import io
import PIL

import common
from jit import core as jit_core
from jit import maths as jit_maths
from scene import Scene

NUM_SAMPLES = 1 # number of sample per pixel

@numba.njit
def shade(ray : jit_core.Ray, hit : jit_core.Hit):
    if hit.valid():
        dot =  math.fabs(jit_maths.dot(ray.d, hit.n))
        return hit.diffuse * dot
    # background colour
    return np.asarray([10,10,10])/255.0

@numba.njit
def trace(ray : jit_core.Ray, details):
    hit = jit_maths.intersect(ray, details)
    return shade(ray, hit)

@common.timeit
@numba.njit
def render(image, camera, details):
    for i in range(camera.width):
        for j in range(camera.height):
            for _ in range(NUM_SAMPLES):
                ray = camera.ray(i, j)
                image[camera.height-1-j, i] = trace(ray, details)

@common.timeit
def show(image):
    buffer = io.BytesIO()
    PIL.Image.fromarray(np.uint8(image*255)).save(buffer, 'png')
    IPython.display.display(IPython.display.Image(data=buffer.getvalue()))

def main():
    scene = Scene()
    scene.load_cornell_box()
    details = scene.details()
    camera = scene.camera
    camera.set_resolution(640, 480)
    image = np.empty((camera.height, camera.width, 3))
    render(image, camera, details)
    show(image)

if __name__ == '__main__':
    main()

