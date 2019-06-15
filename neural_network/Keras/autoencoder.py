"""
@author: Vincent Bonnet
@description : Keras Evaluation - Image Autoencoder
"""

from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

def get_test_and_training_images():
    '''
    Returns test data
    '''
    (x_train, _), (x_test, _) = mnist.load_data()

    # Normalize Data
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # Flatten 2D images into 1D array
    # x_train.shape = (num_images, width, height) to (num_image, width*height)
    # x_test.shape = (num_images, width, height) to (num_image, width*height)
    image_shape = x_train.shape[1:]
    total_pixel = np.prod(image_shape)

    num_training = x_train.shape[0]
    num_test = x_test.shape[0]

    x_train = x_train.reshape(num_training, total_pixel)
    x_test = x_test.reshape(num_test, total_pixel)

    return x_train, x_test

def show_images():
    '''
    Show images
    '''
    pass

def get_autoencoder():
    '''
    Returns compiled Neural Network Model
    '''
    # Build the layers
    input_layer = Input(shape=(784,))
    encoded_layer = Dense(units=128, activation='relu')(input_layer)
    encoded_layer = Dense(units=64, activation='relu')(encoded_layer)
    encoded_layer = Dense(units=32, activation='relu')(encoded_layer)
    decoded_layer = Dense(units=64, activation='relu')(encoded_layer)
    decoded_layer = Dense(units=128, activation='relu')(decoded_layer)
    decoded_layer = Dense(units=784, activation='sigmoid')(decoded_layer)

    # Build the model
    autoencoder = Model(input_layer, decoded_layer)
    #encoder = Model(input_layer, encoded_layer)
    #autoencoder.summary()
    #encoder.summary()

    # Compile model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return autoencoder


def main():
    '''
    Execute training and test it
    '''
    auto_encoder = get_autoencoder()
    x_train, x_test = get_test_and_training_images()

if __name__ == '__main__':
    main()
