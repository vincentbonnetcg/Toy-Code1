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

NUM_EXAMPLES_IN_DATASET = 500
KERAS_OUTPUT_FOLDER = 'keras_output/'

def export_model_to_png(model):
    os.makedirs(os.path.dirname(KERAS_OUTPUT_FOLDER), exist_ok=True)
    plot_model(model, to_file=KERAS_OUTPUT_FOLDER + 'model.png', show_shapes=True, show_layer_names=True)

def load_example(file_id):
    dataset_folder = os.path.join(os.path.dirname(__file__), 'dataset')
    file_path = os.path.join(dataset_folder, 'file' + str(file_id) + '.npz')

    if not os.path.exists(file_path):
        return None

    npzfile = np.load(file_path)
    bone_infos = npzfile['bone_infos']
    rigid_skinning = npzfile['rigid_skinning']
    smooth_skinning = npzfile['smooth_skinning']
    return bone_infos, rigid_skinning, smooth_skinning

def normalize(data):
    min_v = np.min(data)
    max_v = np.max(data)
    data -= min_v
    data /= (max_v - min_v)

def set_data(x, y, file_ids):
    for i, file_id in enumerate(file_ids):
        bone_infos, rigid_skinning, smooth_skinning = load_example(file_id+1)
        x[i][:] = bone_infos.flatten()[:]
        offsets = smooth_skinning - rigid_skinning
        y[i][:] = offsets.flatten()[:]

def load_dataset(percentage_test_from_dataset = 0.1):
    # compute input and output shapes
    bone_infos, rigid_skinning, smooth_skinning = load_example(file_id=1)
    in_shape = bone_infos.flatten().shape[0]
    out_shape = smooth_skinning.flatten().shape[0]

    # pre-allocate tests and training data
    example_ids = np.arange(NUM_EXAMPLES_IN_DATASET)
    np.random.shuffle(example_ids)
    num_test = int(NUM_EXAMPLES_IN_DATASET * percentage_test_from_dataset)
    num_train = NUM_EXAMPLES_IN_DATASET - num_test
    test_ids = example_ids[num_train:]
    train_ids = example_ids[:num_train]

    x_train = np.empty((num_train, in_shape))
    y_train = np.empty((num_train, out_shape))
    x_test = np.empty((num_test, in_shape))
    y_test = np.empty((num_test, out_shape))

    # set data
    set_data(x_train, y_train, train_ids)
    set_data(x_test, y_test, test_ids)

    # normalize data
    # TODO - add container to store the re-scaling parameters
    #normalize(x_train)
    #normalize(y_train)

    return x_train, y_train, x_test, y_test

def main():
    x_train, y_train, x_test, y_test = load_dataset()
    in_shape = x_train.shape[1]
    out_shape = y_train.shape[1]

    # Create NN model
    input_layer = Input(shape=(in_shape,))
    layer_0 = Dense(units=32, activation='tanh', name='HiddenLayer0')(input_layer)
    layer_1 = Dense(units=128, activation='tanh', name='HiddenLayer1')(layer_0)
    layer_2 = Dense(units=out_shape, activation='tanh', name='HiddenLayer2')(layer_1)
    model = Model(input_layer, layer_2)

    optimizer = keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = keras.losses.mean_squared_error
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    export_model_to_png(model)

    # Train data
    model.fit(x=x_train, y=y_train,
              epochs=100,
              batch_size=64,
              shuffle=True,
              validation_data=(x_test, y_test))

    # Predict
    print(np.min(y_train), np.max(y_train))
    x_test = x_train[10:11]
    predicted = model.predict(x_test)
    diff = y_train[10]-predicted[0]
    print(np.min(diff), np.max(diff))

if __name__ == '__main__':
    main()

