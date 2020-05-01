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

'''
 Global Parameters
'''
NUM_NODES = 128 # For GPU should be a multiplier of TPB
JACOBI_ITERATIONS = 10000
TPB = 16 # Thread per block
GPU_SHARED_MEMORY_SIZE = TPB + 2 # +2 because the stencil requires -1 and +1

def create_domain(num_nodes=64):
    '''
    Returns domain nodes along x, y and grid values
    The values come from an image
    '''
    # Convert image to greyscale numpy 2D-array
    image = skimage.img_as_float(skimage.color.rgb2gray(skimage.data.chelsea())).astype(np.float32)
    # Resize the image to (num_nodes, num_nodes) shape
    image = skimage.transform.resize(image, output_shape=(num_nodes, num_nodes), anti_aliasing=True)
    # Flip the image upside-down
    image = np.flipud(image)
    # Finally, turn the image into a single memory block to be used by Cuda
    image = np.ascontiguousarray(image, dtype=np.float32)

    return image

def create_mask(num_nodes=64):
    values = np.random.rand(num_nodes, num_nodes)
    ratio_of_zero = 0.4 # 10 % dark area
    values[values > ratio_of_zero] = 1.0
    values[values <= ratio_of_zero] = 0.0
    values[0][:] = 1.0 # top row
    values[-1][:] = 1.0 # bottom row
    values[:,0] = 1.0 # left column
    values[:,-1] = 1.0 # right column
    values[40:54, 40:50]= 0.0 # extra dark area for some reasons
    values[80:95, 65:75]= 0.0 # extra dark area for no reason

    return values

@numba.njit(parallel=True)
def numba_jacobi_solver_with_mask(x, next_x, mask_indices):
    num_mask_indices = len(mask_indices)

    for it in numba.prange(num_mask_indices):
        mask_index = mask_indices[it]
        i = mask_index[0]
        j = mask_index[1]
        next_x[i][j] = (x[i-1][j] + x[i+1][j] + x[i][j-1] + x[i][j+1]) * 0.25

def laplace_inpainting(domain, mask, num_iterations):
    buffers = [domain, np.copy(domain)]
    for iteration in range(num_iterations):
        numba_jacobi_solver_with_mask(buffers[0], buffers[1], mask_indices)
        buffers[0], buffers[1] = buffers[1], buffers[0] # swap buffers

    # show result
    fig, ax = plt.subplots()
    domain_points = np.linspace(0, 10, num=NUM_NODES, endpoint=True)
    im = ax.pcolormesh(domain_points, domain_points, buffers[0], cmap="gist_gray", antialiased=True, shading="gouraud")
    fig.colorbar(im)
    #fig.savefig("test.png")

if __name__ == '__main__':
    # create domain
    domain = create_domain(NUM_NODES)
    mask_values = create_mask(NUM_NODES)
    domain *= mask_values
    mask_indices = np.argwhere(mask_values == 0)
    laplace_inpainting(domain, mask_indices, JACOBI_ITERATIONS)



