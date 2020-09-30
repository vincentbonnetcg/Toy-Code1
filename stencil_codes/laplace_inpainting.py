"""
@author: Vincent Bonnet
@description : Image recovery by solving Laplace's Equation on CPU
    Jacobi method solves Ax=b where
    A contains the coefficient of the discrete laplace operator
    0 -1  0
    -1 4 -1
    0 -1  0
    x is the unknown discretized function (array)
    b is equal to zero
    By definition, The jacobi iteration is :
    xi(k+1) = 1/aii * (bi - sum(aij * xj(k)) 'where j!=i')
    because b is a zero array and aii reference the coefficient 4
    xi(k+1) = 1/4 * (- sum(aij * xj(k)) 'for j!=i')
    and aij are -1 for j!=i
    => xi(k+1) = 1/4 * (sum(xj(k)) 'for j!=i')
"""

import numba
import numpy as np
import matplotlib.pyplot as plt
import skimage

JACOBI_ITERATIONS = 1000
RATIO_OF_ZERO = 0.95 # 0. => original image , 1. < just zeros

def create_mask(shape):
    values = np.random.rand(shape[0], shape[1])
    values[values > RATIO_OF_ZERO] = 1.0
    values[values <= RATIO_OF_ZERO] = 0.0
    values[0][:] = 1.0 # top row
    values[-1][:] = 1.0 # bottom row
    values[:,0] = 1.0 # left column
    values[:,-1] = 1.0 # right column
    return values

@numba.njit(parallel=True)
def numba_jacobi_solver_with_mask(x, next_x, mask_indices):
    num_mask_indices = len(mask_indices)

    for it in numba.prange(num_mask_indices):
        mask_index = mask_indices[it]
        i = mask_index[0]
        j = mask_index[1]
        next_x[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1]) * 0.25

def laplace_inpainting(image, mask_indices, num_iterations):
    buffers = [image, np.copy(image)]
    for iteration in range(num_iterations):
        numba_jacobi_solver_with_mask(buffers[0], buffers[1], mask_indices)
        buffers[0], buffers[1] = buffers[1], buffers[0] # swap buffers

if __name__ == '__main__':
    # load image
    image = skimage.img_as_float(skimage.color.rgb2gray(skimage.data.chelsea()))
    img = plt.imshow(image)
    img.set_cmap('gray')
    plt.title('Original image')
    plt.show()

    # remove data from image
    mask = create_mask(image.shape)
    image *= mask
    img = plt.imshow(image)
    img.set_cmap('gray')
    plt.title('Missing data')
    plt.show()

    # recover image with laplace inpainting
    mask_indices = np.argwhere(mask == 0)
    laplace_inpainting(image, mask_indices, JACOBI_ITERATIONS)
    img = plt.imshow(image)
    img.set_cmap('gray')
    plt.title('Laplace inpainting')
    plt.show()
