"""
@author: Vincent Bonnet
@description : Pathtracer with Python+Numba
"""

import IPython.display

import numpy as np
import io
import PIL

import common
from scene import Scene
import jit.pathtracer as pathtracer

NUM_SAMPLES = 1 # number of sample per pixel

@common.timeit
def force_jit(image, camera, details):
    # jit compilation by calling a tiny scene
    width, height = camera.width, camera.height
    camera.set_resolution(2, 2)
    pathtracer.render(image, camera, details, NUM_SAMPLES)
    camera.set_resolution(width, height)

@common.timeit
def render(image, camera, details):
    pathtracer.render(image, camera, details, NUM_SAMPLES)

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
    camera.set_resolution(512, 512)
    image = np.zeros((camera.height, camera.width, 3))

    force_jit(image, camera, details)
    render(image, camera, details)
    show(image)

if __name__ == '__main__':
    main()

