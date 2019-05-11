"""
@author: Vincent Bonnet
@description : Neural Network main
"""

import data_loader
import neural_network as nn
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 1.0
MAX_EPOCH = 1000

def draw_sample_mnist(loader_mnist):
    mnist_data = loader_mnist.load_into_array()
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
    plt.imshow(test_image, cmap='gray')

def main():
    # Download and collect the numpy array
    loader_mnist = data_loader.MNIST_Loader()
    loader_mnist.download()
    mnist_data = loader_mnist.load_into_array()

    # Draw sample of MNIST data
    #draw_sample_mnist(loader_mnist)

    # Create logistic regression with its Hyperparameters
    number_to_detect = 0
    neural_network = nn.LogisticRegressionMNIST(number_to_detect)
    neural_network.learning_rate = LEARNING_RATE
    neural_network.num_epoch = MAX_EPOCH

    # Train
    neural_network.train(mnist_data["training_images"])



if __name__ == '__main__':
    main()
