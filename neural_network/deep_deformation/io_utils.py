"""
@author: Vincent Bonnet
@description : Manage read/write for the various files/folder
"""
import numpy as np
import os

DATASET_FOLDER = 'dataset'
PREDICTION_FOLDER = 'prediction'

def get_dataset_folder():
    return os.path.join(os.path.dirname(__file__), DATASET_FOLDER)

def get_prediction_folder():
    return os.path.join(os.path.dirname(__file__), PREDICTION_FOLDER)

def load_single_dataset(dataset_file):
    dataset_folder = get_dataset_folder()
    file_path = os.path.join(dataset_folder, dataset_file)

    if not os.path.exists(file_path):
        return None

    npzfile = np.load(file_path)
    return npzfile['bones'],  npzfile['bases'], npzfile['smooths']

def normalize(data):
    min_v = np.min(data)
    max_v = np.max(data)
    data -= min_v
    data /= (max_v - min_v)

def load_dataset(dataset_file, test_ratio = 0.1):
    # load a single dataset
    # TODO : should load from multiple datasets !
    bones, bases, smooths = load_single_dataset(dataset_file)
    num_examples = len(bones)

    in_shape = np.prod(bones.shape[1:])
    out_shape = np.prod(smooths.shape[1:])

    # pre-allocate tests and training data
    example_ids = np.arange(num_examples)
    np.random.shuffle(example_ids)
    num_test = int(num_examples * test_ratio)
    num_train = num_examples - num_test
    test_ids = example_ids[num_train:]
    train_ids = example_ids[:num_train]

    x_train = np.empty((num_train, in_shape))
    y_train = np.empty((num_train, out_shape))
    x_test = np.empty((num_test, in_shape))
    y_test = np.empty((num_test, out_shape))

    # set data for training and test
    for i, example_id in enumerate(train_ids):
        x_train[i][:] = bones[example_id].flatten()[:]
        y_train[i][:] = smooths[example_id].flatten()[:] - bases[example_id].flatten()[:]

    for i, example_id in enumerate(test_ids):
        x_test[i][:] = bones[example_id].flatten()[:]
        y_test[i][:] = smooths[example_id].flatten()[:] - bases[example_id].flatten()[:]

    # normalize data
    # TODO - add container to store the re-scaling parameters
    #normalize(x_train)
    #normalize(y_train)
    #normalize(x_test)
    #normalize(y_test)

    return x_train, y_train, x_test, y_test, train_ids, test_ids


def write_predicted_dataset(dataset_file, predicted_offsets, example_ids):
    predict_folder = get_prediction_folder()
    if not os.path.exists(predict_folder):
        os.makedirs(predict_folder)

    file_path = os.path.join(predict_folder, dataset_file)
    if not os.path.exists(file_path):
        bones, bases, smooths = load_single_dataset(dataset_file)
        predicted = np.copy(bases)

        for i, example_id in enumerate(example_ids):
            offsets = predicted_offsets[i].reshape(predicted.shape[1:])
            predicted[example_id] = bases[example_id] + offsets

        out_attributes = {'bones' : bones,
                          'bases' : bases,
                          'smooths' : smooths,
                          'predicted_smooths' : predicted}
        np.savez(file_path, **out_attributes)
    else:
        npzfile = np.load(file_path)
        bones = npzfile['bones']
        bases = npzfile['bases']
        smooths = npzfile['smooths']
        predicted = npzfile['predicted_smooths']

        for i, example_id in enumerate(example_ids):
            offsets = predicted_offsets[i].reshape(predicted.shape[1:])
            predicted[example_id] = bases[example_id] + offsets

        out_attributes = {'bones' : bones,
                          'bases' : bases,
                          'smooths' : smooths,
                          'predicted_smooths' : predicted}
        np.savez(file_path, **out_attributes)



