"""
@author: Vincent Bonnet
@description : Pathtracer with Python+Numba
"""

import IPython.display

import numpy as np
import io
import PIL
import time
from concurrent.futures import ThreadPoolExecutor

import common
from scene import Scene
import jit.pathtracer as pathtracer

CPU_COUNT = 6

@common.timeit
def force_jit(image, camera, details):
    # jit compilation by calling a tiny scene
    width, height = camera.width, camera.height
    camera.set_resolution(2, 2)
    pathtracer.render(image, camera, details, time.time())
    camera.set_resolution(width, height)

@common.timeit
def render_MT(image, camera, details):
    row_start = 0
    row_step = CPU_COUNT
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        start_time = time.time()
        for thread_id in range(CPU_COUNT):
            row_start = thread_id
            executor.submit(pathtracer.render, image, camera, details, start_time,
                            row_start, row_step)

@common.timeit
def render(image, camera, details):
    start_time = time.time()
    pathtracer.render(image, camera, details, start_time)

@common.timeit
def show(image):
    buffer = io.BytesIO()
    PIL.Image.fromarray(np.uint8(image*255)).save(buffer, 'png')
    IPython.display.display(IPython.display.Image(data=buffer.getvalue()))

def main():
    pathtracer.MAX_DEPTH = 10 # max ray bounces
    pathtracer.NUM_SAMPLES = 50 # number of sample per pixel
    pathtracer.RANDOM_SEED = 10

    scene = Scene()
    scene.load_cornell_box()
    details = scene.tri_details()
    camera = scene.camera
    camera.set_resolution(512, 512)
    image = np.zeros((camera.height, camera.width, 3))

    force_jit(image, camera, details)
    render_MT(image, camera, details)
    show(image)

if __name__ == '__main__':
    main()
