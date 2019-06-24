"""
@author: Vincent Bonnet
@description : Keras Evaluation - Variational Autoencoder
"""

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import keras_utils

def get_variational_autoencoder():
    '''
    Returns compiled Neural Network Model
    '''
    # Build the layers
    input_layer = Input(shape=(784,))
    encoded_layer = Dense(units=128, activation='relu', name='EncoderLayer1')(input_layer)
    encoded_layer = Dense(units=64, activation='relu', name='EncoderLayer2')(encoded_layer)
    #mean_layer = Dense(name='Means')(encoded_layer)
    #standard_deviation_layer = Dense(name='StandardDeviation')(encoded_layer)
    encoded_layer = Dense(units=32, activation='relu', name='SampleLayer')(encoded_layer)
    decoded_layer = Dense(units=64, activation='relu', name='DecoderLayer1')(encoded_layer)
    decoded_layer = Dense(units=128, activation='relu', name='DecoderLayer2')(decoded_layer)
    decoded_layer = Dense(units=784, activation='sigmoid', name='Output')(decoded_layer)

    # Build the model
    autoencoder = Model(input_layer, decoded_layer)
    #autoencoder.summary()

    # Compile model
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return autoencoder


def main():
    '''
    Execute training and test the neural network
    '''
    # Get NN model and data
    variational_autoencoder = get_variational_autoencoder()
    x_train, x_test = keras_utils.get_MNIST_test_and_training_data()

    # Plot Informations
    keras_utils.plot_model_to_file(variational_autoencoder)

    keras_utils.save_weights(variational_autoencoder)

if __name__ == '__main__':
    main()


