"""
@author: Vincent Bonnet
@description : Keras Evaluation - Variational Autoencoder
see : https://keras.io/examples/variational_autoencoder/
"""

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import keras_utils

LATENT_DIM = 2
EPOCH = 10

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_variational_autoencoder():
    '''
    Returns compiled Neural Network Model
    '''
    def get_encoder():
        '''
        Build the encoder layers
        From 784 to 2
        '''
        input_layer = Input(shape=(784,))
        encoder_layer = Dense(units=128, activation='relu', name='encoderLayer1')(input_layer)
        encoder_layer = Dense(units=64, activation='relu', name='encoderLayer2')(encoder_layer)

        z_mean_layer = Dense(units=LATENT_DIM, name='z_mean')(encoder_layer)
        z_log_var_layer  = Dense(units=LATENT_DIM, name='z_log_var')(encoder_layer)
        z_layer = Lambda(sampling, output_shape=(LATENT_DIM,), name='z')([z_mean_layer, z_log_var_layer])

        encoder = Model(input_layer, [z_mean_layer, z_log_var_layer, z_layer], name='encoder')
        return encoder, input_layer

    def get_decoder():
        '''
        Build the decode layers
        From 2 to 784
        '''
        latent_inputs = Input(shape=(LATENT_DIM,), name='z_sampling')
        decoder_layer = Dense(units=64, activation='relu', name='decoderLayer1')(latent_inputs)
        output_layer = Dense(units=128, activation='sigmoid', name='decoderLayer2')(decoder_layer)

        decoder = Model(latent_inputs, output_layer, name='decoder')
        return decoder

    # Build the model
    encoder, input_layer = get_encoder()
    decoder = get_decoder()

    vae_output_layer = decoder(encoder(input_layer)[2])
    vae = Model(input_layer, vae_output_layer, name='vae')
    return vae

def main():
    '''
    Execute training and test the neural network
    '''
    # Get NN model and data
    variational_autoencoder = get_variational_autoencoder()
    x_train, x_test = keras_utils.get_test_and_training_data('mnist')

    # Plot Informations
    keras_utils.plot_model_to_file(variational_autoencoder)

    keras_utils.save_weights(variational_autoencoder)

if __name__ == '__main__':
    main()
