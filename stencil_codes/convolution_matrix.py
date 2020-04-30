"""
@author: Vincent Bonnet
@description : convolution matrix on GPU
"""

import numpy as np
import numba
import math
import skimage
import img_utils

@numba.vectorize(['float32(float32, float32)'], target='cuda')
def combine_images(a, b):
    return max(min(a+b, 1.0), 0.0)

@numba.cuda.jit
def apply_kernel(image, imageResult, Gx):
    #x = numba.cuda.threadIdx.x + (cuda.blockIdx.x * cuda.blockDim.x)
    #y = numba.cuda.threadIdx.y + (cuda.blockIdx.y * cuda.blockDim.y)
    x, y = numba.cuda.grid(2)
    value = 0.0
    for i in range(-1, 2):
        for j in range(-1, 2):
            xi = x + i
            yi = y + j
            if xi >= 0 and yi >= 0 and xi < image.shape[0] and yi < image.shape[1]:
                value += (image[xi, yi] * Gx[i, j])

    imageResult[x, y] = value

def image_processing_scene(image, kernels):
    # Setup images
    images = [None, None]
    images[0] = image.copy()
    images[1] = image.copy()

    # Setup blocks
    threadsPerBlock = (16, 16)
    blocksPerGridX = math.ceil(images[0].shape[0] / threadsPerBlock[0])
    blocksPerGridY = math.ceil(images[0].shape[1] / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGridX, blocksPerGridY)

    # Run kernel on Cuda
    apply_kernel[blocksPerGrid, threadsPerBlock](images[0], images[1], kernels[0])
    apply_kernel[blocksPerGrid, threadsPerBlock](images[1], images[0], kernels[1])

    # Show Result
    skimage.io.imshow(combine_images(image, images[0]))

if __name__ == '__main__':
    # create image
    image = skimage.img_as_float(skimage.color.rgb2gray(skimage.data.chelsea())).astype(np.float32)
    image = img_utils.resize_image_and_keep_ratio(image, 512, 512)
    # Common Kernels
    sobelYKernel = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    gaussianBlurKernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]) / 16
    '''
    sobelXKernel = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    identityKernel = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    sharpenKernel = np.array([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]])
    embossKernel = np.array([[-2.0, -1.0, 0.0], [-1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])
    '''
    # Execute
    image_processing_scene(image, [sobelYKernel, gaussianBlurKernel])

