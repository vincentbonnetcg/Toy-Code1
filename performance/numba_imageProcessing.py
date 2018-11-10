"""
@author: Vincent Bonnet
@description : Evaluate GPU capabilities with Python+Numba by trying image processing
               Efficient 2D Stencil Computations Using CUDA
"""

import math
import numpy as np
from numba import cuda, vectorize
from skimage import data, io, color, transform, feature
import skimage as skimage
import matplotlib.pyplot as plt

@vectorize(['float32(float32, float32)'], target='cuda')
def combine_images(a, b):
    return max(min(a+b, 1.0), 0.0)

@cuda.jit
def apply_kernel(image, imageResult, Gx):
    #x = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    #y = cuda.threadIdx.y + (cuda.blockIdx.y * cuda.blockDim.y)
    x, y = cuda.grid(2)
    value = 0.0
    for i in range(-1, 2):
        for j in range(-1, 2):
            xi = x + i
            yi = y + j
            if xi >= 0 and yi >= 0 and xi < image.shape[0] and yi < image.shape[1]:
                value += (image[xi, yi] * Gx[i, j])

    imageResult[x, y] = value

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

# Resize an image while keeping its ratio
def resize_image_and_keep_ratio(image, width, height):
    out = np.zeros((width, height), dtype=np.float32)
    scaleX = image.shape[0] / width
    scaleY = image.shape[1] / height
    maxScale = max(scaleX, scaleY)
    newWidth = np.int(image.shape[0] / maxScale)
    newHeight = np.int(image.shape[1] / maxScale)
    tmpImage = transform.resize(image, (newWidth, newHeight))
    offsetX = np.int((width - tmpImage.shape[0]) / 2)
    offsetY = np.int((height - tmpImage.shape[1]) / 2)
    out[offsetX:offsetX+tmpImage.shape[0], offsetY:offsetY+tmpImage.shape[1]] = tmpImage
    return out

# Tests Functions
def image_processing_scene():
    # Setup images
    images = [None, None]
    image = skimage.img_as_float(color.rgb2gray(data.chelsea())).astype(np.float32)
    image = resize_image_and_keep_ratio(image, 512, 512)
    images[0] = image.copy()
    images[1] = image.copy()

    # Setup blocks
    threadsPerBlock = (16, 16)
    blocksPerGridX = math.ceil(images[0].shape[0] / threadsPerBlock[0])
    blocksPerGridY = math.ceil(images[0].shape[1] / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGridX, blocksPerGridY)

    # Common Kernels
    #sobelXKernel = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    sobelYKernel = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    #identityKernel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    #sharpenKernel = np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]])
    #embossKernel = np.array([[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    gaussianBlurKernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16

    # Run kernel on Cuda
    apply_kernel[blocksPerGrid, threadsPerBlock](images[0], images[1], sobelYKernel)
    apply_kernel[blocksPerGrid, threadsPerBlock](images[1], images[0], gaussianBlurKernel)

    # Show Result
    io.imshow(combine_images(image, images[1]))


def conways_game_of_life_scene(numIterations, renderFolder):
    # Setup images
    images = [None, None]
    images[0] = skimage.img_as_float(color.rgb2gray(data.chelsea())).astype(np.float32)
    images[0] = resize_image_and_keep_ratio(images[0], 128, 128)
    images[0] = feature.canny(images[0], sigma=1)
    images[1] = images[0].copy()

    # Setup blocks
    threadsPerBlock = (16, 16)
    blocksPerGridX = math.ceil(images[0].shape[0] / threadsPerBlock[0])
    blocksPerGridY = math.ceil(images[0].shape[1] / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGridX, blocksPerGridY)

    # Run kernel on Cuda and show results
    id0 = True
    for i in range(1, numIterations):
        id0 = not id0 # buffer id to process
        id1 = not id0 # buffer id to hold the result
        fig = plt.figure()
        io.imshow(images[id0])
        if i > 10:
            apply_cellular_automata_rules[blocksPerGrid, threadsPerBlock](images[int(id0)], images[int(id1)])
        if (len(renderFolder)):
            filename = str(i).zfill(4) + " .png"
            fig.savefig(renderFolder + "/" + filename)


# Run Tests
#image_processing_scene()
conways_game_of_life_scene(50, "")
