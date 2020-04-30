"""
@author: Vincent Bonnet
@description : Game of life on GPU
"""

import math
import numpy as np
from numba import cuda
from skimage import data, io, color, feature
import skimage as skimage
import matplotlib.pyplot as plt
import img_utils

RENDER_FOLDER = ""

@cuda.jit
def apply_cellular_automata_rules(image, imageResult):
    #x = cuda.threadIdx.x + (cuda.blockId x.x * cuda.blockDim.x)
    #y = cuda.threadIdx.y + (cuda.blockIdx.y * cuda.blockDim.y)
    x, y = cuda.grid(2)
    numNeighbours = 0
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i != 0 or j != 0:
                xi = x + i
                yi = y + j
                if xi >= 0 and yi >= 0 and xi < image.shape[0] and yi < image.shape[1] and image[xi, yi] == 1.0:
                    numNeighbours += 1

    imageResult[x, y] = image[x, y]
    if imageResult[x, y] == 1.0: # live cell ...
        if numNeighbours < 2: # under population.
            imageResult[x, y] = 0.0
        elif numNeighbours > 3: # overpopulation
            imageResult[x, y] = 0.0
    else: # dead cell ...
        if numNeighbours == 3: # reproduction
            imageResult[x, y] = 1.0


def conways_game_of_life_scene(image, iterations):
    # Setup images
    images = [None, None]
    images[0] = image
    images[0] = feature.canny(images[0], sigma=1)
    images[1] = images[0].copy()

    # Setup blocks
    threadsPerBlock = (16, 16)
    blocksPerGridX = math.ceil(images[0].shape[0] / threadsPerBlock[0])
    blocksPerGridY = math.ceil(images[0].shape[1] / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGridX, blocksPerGridY)

    # Run kernel on Cuda and show results
    id0 = True

    for i in range(1, iterations):
        id0 = not id0 # buffer id to process
        id1 = not id0 # buffer id to hold the result
        fig = plt.figure()
        io.imshow(images[id0])
        if i > 10:
            apply_cellular_automata_rules[blocksPerGrid, threadsPerBlock](images[int(id0)], images[int(id1)])
        if (len(RENDER_FOLDER)):
            filename = str(i).zfill(4) + " .png"
            fig.savefig(RENDER_FOLDER + "/" + filename)

if __name__ == '__main__':
    image = skimage.img_as_float(color.rgb2gray(data.chelsea())).astype(np.float32)
    image = img_utils.resize_image_and_keep_ratio(image, 128, 128)

    conways_game_of_life_scene(image, 50)
