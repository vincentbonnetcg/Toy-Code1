"""
@author: Vincent Bonnet
@description : Keras Evaluation - Variational Autoencoder
see : https://keras.io/examples/variational_autoencoder/
"""

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras.losses import binary_crossentropy
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import keras_utils

ORIGIN_DIM = 784
LATENT_DIM = 2
HIDDEN_DIM = 521
EPOCH = 50
BATCH_SIZE = 50
TRAINING_MODE = False # True (training), False (prediction)

def sampling_z(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0] # should be BATCH_SIZE
    dim = K.int_shape(z_mean)[1] # should be LATENT_DIM
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def get_variational_autoencoder():
    '''
    Returns compiled Neural Network Model
    '''
    # Decoder Layers (From ORIGIN_DIM to 2)
    input_layer = Input(shape=(ORIGIN_DIM,))
    encoder_layer = Dense(units=HIDDEN_DIM, activation='relu', name='encoder_layer')(input_layer)

    z_mean_layer = Dense(units=LATENT_DIM, activation = 'linear', name='z_mean')(encoder_layer)
    z_log_var_layer  = Dense(units=LATENT_DIM, activation = 'linear', name='z_log_var')(encoder_layer)
    z_layer = Lambda(sampling_z, output_shape=(LATENT_DIM,), name='z')([z_mean_layer, z_log_var_layer])

    # Encoder Layers (From 2 to ORIGIN_DIM)
    # From 2 to 784
    decoder_layer = Dense(units=HIDDEN_DIM, activation='relu', name='decoder_layer')(z_layer)
    output_layer = Dense(units=ORIGIN_DIM, activation='sigmoid', name='decoder_out')(decoder_layer)

    # VAE model
    vae = Model(input_layer, output_layer, name='vae')

    # Add loss function to VAE model
    # Shameless straight copy-paste from https://keras.io/examples/variational_autoencoder/
    reconstruction_loss = binary_crossentropy(input_layer, output_layer)
    reconstruction_loss *= ORIGIN_DIM
    kl_loss = 1 + z_log_var_layer - K.square(z_mean_layer) - K.exp(z_log_var_layer)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    return vae

def main():
    '''
    Execute training and test the neural network
    '''
    # Get NN model and data
    vae = get_variational_autoencoder()
    x_train, x_test = keras_utils.get_test_and_training_data('mnist')

    # Plot NN Structure
    keras_utils.plot_model_to_file(vae)

    # Train the network
    if TRAINING_MODE:
        vae.fit(x_train,
                batch_size=BATCH_SIZE,
                nb_epoch=EPOCH,
                shuffle=True,
                validation_data=(x_test, None))

        keras_utils.save_weights(vae)
    else:
        keras_utils.load_weights(vae)

    # TODO : show something


if __name__ == '__main__':
    main()
