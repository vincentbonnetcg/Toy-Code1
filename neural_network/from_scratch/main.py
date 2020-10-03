"""
@author: Vincent Bonnet
@description : Neural Network main
"""

# FROM SCRATCH in main
# http://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%201/
# https://jonathanweisberg.org/post/A%20Neural%20Network%20from%20Scratch%20-%20Part%202/

# https://www.codementor.io/mgalarny/making-your-first-machine-learning-classifier-in-scikit-learn-python-db7d7iqdh

# http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/
# https://iamtrask.github.io/2015/07/12/basic-python-network/

# https://nyilmazdata.net/2018/03/01/hello-world-of-machine-learning/
# https://github.com/valivetiravichandra/mnist-from-scratch-using-numpy

# Interisting animated gif
# https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795


# TODO - add another cost function
# TODO - add bias update 
# TODO - understand the derivation and weight update 
# TODO - serialzie/deserialize the trained network
# TODO - number_to_detect should be part of the training_labels and not neural network
# TODO - how to pick the result ? statistics ? 
# https://www.youtube.com/watch?v=FTr3n7uBIuE  - convolution neural network


import data_loader
import neural_network as nn
import matplotlib.pyplot as plt
import numpy as np

LEARNING_RATE = 100.0
MAX_EPOCH = 1

def draw_sample_mnist(loader_mnist):
    mnist_data = loader_mnist.load_into_float_array(normalize = False) # Get values from 0-255
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
    mnist_data = loader_mnist.load_into_float_array(normalize = True) # Get values from 0-1

    # Draw sample of MNIST data
    #draw_sample_mnist(loader_mnist)

    # Create logistic regression with its Hyperparameters
    number_to_detect = 2.0
    neural_network = nn.LogisticRegressionMNIST(number_to_detect)
    neural_network.learning_rate = LEARNING_RATE
    neural_network.num_epoch = MAX_EPOCH

    # Training and Tests
    neural_network.train(mnist_data["training_images"], mnist_data["training_labels"])
    neural_network.test(mnist_data["test_images"], mnist_data["test_labels"])


if __name__ == '__main__':
    main()
