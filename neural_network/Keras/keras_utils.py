"""
@author: Vincent Bonnet
@description : Utilities for Keras
"""

from keras.utils import plot_model
from keras.datasets import mnist, fashion_mnist
import numpy as np
import os

KERAS_OUTPUT_FOLDER = "keras_output/"
KERAS_MODEL_PLOT_FILE = "model.png"
KERAS_ARCHITECTURE_FILE = "model.json"
KERAS_MODEL_WEIGHTS_FILE = "weights.h5"


def prepare_Keras_folder():
    '''
    Create a folder to store histograms, trained models etc.
    '''
    os.makedirs(os.path.dirname(KERAS_OUTPUT_FOLDER), exist_ok=True)

def plot_model_to_file(model):
    '''
    Export model diagram into a file
    '''
    prepare_Keras_folder()
    plot_model(model, to_file=KERAS_OUTPUT_FOLDER + KERAS_MODEL_PLOT_FILE, show_shapes=True, show_layer_names=True)

def save_weights(model):
    '''
    Save model and weight to into files
    '''
    prepare_Keras_folder()
    model.save_weights(KERAS_OUTPUT_FOLDER + KERAS_MODEL_WEIGHTS_FILE)

def get_test_and_training_data(data_name):
    '''
    Returns test data
    '''
    if data_name == 'mnist':
        (x_train, _), (x_test, _) = mnist.load_data()
    elif data_name == 'fashion_mnist':
        (x_train, _), (x_test, _) = fashion_mnist.load_data()
    else:
        return None

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
