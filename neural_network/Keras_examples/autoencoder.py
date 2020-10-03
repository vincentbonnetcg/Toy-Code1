"""
@author: Vincent Bonnet
@description : Keras Evaluation - Image Autoencoder
"""

from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import keras_utils

def show_images(test_data, predicted_data):
    '''
    Show images
    '''
    font = {'family': 'arial',
            'color':  'darkblue',
            'weight': 'normal',
            'size': 16 }

    # Original image
    plt.figure(figsize=(15, 4))
    plt.gray()
    for i in range(10):
        ax = plt.subplot(3,  20, i + 1)
        ax.axis('off')
        plt.imshow(test_data[i].reshape(28, 28))
    plt.title('Test Data', fontdict=font)
    plt.show()

    # Predicted image
    plt.figure(figsize=(15, 4))
    plt.gray()
    for i in range(10):
        ax = plt.subplot(3, 20, 2*20 +i+ 1)
        ax.axis('off')
        plt.imshow(predicted_data[i].reshape(28, 28))
    plt.title('Predicted Data', fontdict=font)
    plt.show()


def get_autoencoder():
    '''
    Returns compiled Neural Network Model
    '''
    # Build the layers
    input_layer = Input(shape=(784,))
    encoded_layer = Dense(units=128, activation='relu', name='EncoderLayer1')(input_layer)
    encoded_layer = Dense(units=64, activation='relu', name='EncoderLayer2')(encoded_layer)
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
    Execute training and test it
    '''
    # Get NN model and data
    auto_encoder = get_autoencoder()
    x_train, x_test = keras_utils.get_test_and_training_data('fashion_mnist')

    # Train the autoencoder (input_x==input_y)
    auto_encoder.fit(x=x_train, y=x_train,
                     epochs=10,
                     batch_size=256,
                     shuffle=True,
                     validation_data=(x_test, x_test))

    # Test the autoencoder
    predicted = auto_encoder.predict(x_test)

    # Show results
    show_images(x_test, predicted)
    keras_utils.plot_model_to_file(auto_encoder)


if __name__ == '__main__':
    main()
