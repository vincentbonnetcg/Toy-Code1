"""
@author: Vincent Bonnet
@description : Evaluate GPU capabilities with Python+Numba by trying image processing
               Efficient 2D Stencil Computations Using CUDA
"""

import numpy as np
from numba import cuda, vectorize
from skimage import data, io, color, transform
import skimage as skimage
import math

@vectorize(['float32(float32, float32)'], target='cuda')
def CombineImages(a, b):
    return max(min(a+b, 1.0), 0.0)

@cuda.jit
def applyKernel(image, imageResult, Gx):
    #x = cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    #y = cuda.threadIdx.y + (cuda.blockIdx.y * cuda.blockDim.y)
    x, y = cuda.grid(2)
    value = 0.0
    for i in (range(-1,2)):
        for j in (range(-1,2)):
            xi = x + i
            yi = y + j
            if xi >= 0 and yi >= 0 and xi < image.shape[0] and yi < image.shape[1]:
                value += (image[xi, yi] * Gx[i, j])

    imageResult[x, y] = value

# Resize an image while keeping its ratio
def resizeImageAndKeepRatio(image, width, height):
    out  = np.zeros((width, height), dtype=np.float32)
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

# Create a simple image processing
images = [None, None]
image = skimage.img_as_float(color.rgb2gray(data.chelsea())).astype(np.float32)
image = resizeImageAndKeepRatio(image, 512, 512)
images[0] = image.copy()
images[1] = image.copy()

# Setup blocks
threadsPerBlock = (16, 16)
blocksPerGridX = math.ceil(images[0].shape[0] / threadsPerBlock[0])
blocksPerGridY = math.ceil(images[0].shape[1] / threadsPerBlock[1])
blocksPerGrid = (blocksPerGridX, blocksPerGridY)

# Common Kernels
sobelXKernel = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
sobelYKernel = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
identityKernel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
sharpenKernel = np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]])
embossKernel = np.array([[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
gaussianBlurKernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16

# Run kernel on Cuda
applyKernel[blocksPerGrid, threadsPerBlock](images[0], images[1], sobelYKernel)
applyKernel[blocksPerGrid, threadsPerBlock](images[1], images[0], gaussianBlurKernel)

# Show Result
io.imshow(CombineImages(image, images[1]))
