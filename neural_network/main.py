"""
@author: Vincent Bonnet
@description : main
"""

import data_loader
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Download and collect the numpy arraysc
    loader_mnist = data_loader.MNIST_Loader()
    loader_mnist.download()
    mnist_data = loader_mnist.load_into_array()

    #print(len(mnist_data["test_images"]))

    # Create an image with the first few training_images
    # Only for fun
    num_images_x = 20
    num_images_y = 5
    sx = loader_mnist.size[0]
    sy = loader_mnist.size[1]
    xpixels, ypixels = num_images_x  * sx, num_images_y * sy
    test_image = np.zeros((ypixels, xpixels), dtype=np.uint8)
    image_id = 0
    for i in range(0, num_images_x):
        for j in range(0, num_images_y):
            image_data = mnist_data["test_images"][image_id].reshape(sx,sy)
            test_image[j*sy:j*sy+sy,i*sx:i*sx+sx] = image_data
            image_id += 1

    plt.figure(figsize=(20, 5), dpi=40)
    io.imshow(test_image, cmap='gray')


if __name__ == '__main__':
    main()
