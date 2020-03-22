"""
@author: Vincent Bonnet
@description : Implementation of the deep deformation paper
"""

from keras.utils import plot_model
from keras.layers import Input, Dense
from keras.models import Model
import keras
import numpy as np
import os

#1. load_data split train and test data instead of single one (x_test, x_train, y_test, y_train)
#2. use normalized data
#3. use parameter from paper in optimizer


KERAS_OUTPUT_FOLDER = "keras_output/"
KERAS_MODEL_PLOT_FILE = "model.png"

def export_model_to_file(model):
    '''
    Export model diagram into a folder
    '''
    os.makedirs(os.path.dirname(KERAS_OUTPUT_FOLDER), exist_ok=True)
    plot_model(model, to_file=KERAS_OUTPUT_FOLDER + KERAS_MODEL_PLOT_FILE, show_shapes=True, show_layer_names=True)


def load_single_data(file_ID):
    '''
    Returns the rotation, undeformed and offset data
    '''
    training_path = os.path.dirname(__file__)
    training_path += '/houdini_bridge/training/'

    file_path = 'file' + str(file_ID) + '.npz'
    file_path = training_path + file_path

    offset = None
    undeformed = None
    bone_rotations = None

    if os.path.exists(file_path):
        npzfile = np.load(file_path)
        offset = npzfile['offset']
        undeformed = npzfile['undeformed']
        bone_rotations = npzfile['bone_rotations']

    # flatten numpy
    out_shape = offset.shape[0] * offset.shape[1]
    offset = offset.reshape(out_shape)
    undeformed = undeformed.reshape(out_shape)

    return undeformed, offset, bone_rotations

def load_data():
    num_examples = 400
    undeformed, offset, bone_rotations = load_single_data(file_ID=1)
    in_shape = bone_rotations.shape[0]
    out_shape = offset.shape[0]

    bones = np.empty((num_examples, in_shape))
    offsets = np.empty((num_examples, out_shape))

    offsets[0][:] = offset[:]
    bones[0][:] = bone_rotations[:]

    for file_ID in range(2,num_examples):
        _, offset, bone_rotations = load_single_data(file_ID)
        offsets[file_ID-1][:] = offset[:]
        bones[file_ID-1][:] = bone_rotations[:]

    return undeformed, bones, offsets, in_shape, out_shape

def main():
    # Get data
    undeformed, x_train, y_train, in_shape, out_shape = load_data()

    # Create NN model
    input_layer = Input(shape=(in_shape,))
    layer_0 = Dense(units=32, activation='tanh', name='HiddenLayer0')(input_layer)
    layer_1 = Dense(units=128, activation='tanh', name='HiddenLayer1')(layer_0)
    layer_2 = Dense(units=out_shape, activation='tanh', name='HiddenLayer2')(layer_1)
    model = Model(input_layer, layer_2)

    optimizer = keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = keras.losses.mean_squared_error
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    export_model_to_file(model)

    # Train data
    model.fit(x=x_train, y=y_train,
              epochs=1,
              batch_size=128,
              shuffle=True)
              #validation_data=(x_test, x_test)) # TODO

    # Predict
    x_test = x_train[1:2]
    predicted = model.predict(x_test)
    diff = y_train[1]-predicted[0]
    print(np.min(diff), np.max(diff))

if __name__ == '__main__':
    main()

