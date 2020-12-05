"""
@author: Vincent Bonnet
@description : Implementation of the deep deformation paper
"""

# TODO
#Data preprocessing
  # normalize inputs
  # do not train with bones not involved in the skinning
#Training
  # in load_dataset() should use multiple datasets

import fnmatch,os
import numpy as np
import io_utils
from optimizer import ModelOpt
from skeleton import Skeleton

def training():
    # Get Data
    dataset_file = 'SideFX_Male_walk_startle_L_L_001.npz'
    x_train, y_train, x_test, y_test, train_ids, test_ids = io_utils.load_dataset(dataset_file)
    in_shape = x_train.shape[1]
    out_shape = y_train.shape[1]

    # Get model
    model = ModelOpt()
    model.create_model(in_shape, out_shape)
    # TODO : below should not use test for validation data
    model.set_data(x_train, y_train, x_test, y_test)
    model.fit(epochs=100, batch_size=10)

    # Predict from test
    predicted = model.predict(x_test)
    error = np.sqrt((y_test-predicted)**2)
    print('---- errors ---- ')
    print(np.min(error), np.max(error), np.average(error))
    io_utils.write_predicted_dataset(dataset_file, predicted, test_ids)

    # Predicted from train
    io_utils.write_predicted_dataset(dataset_file, model.predict(x_train), train_ids)

def dataset_summary():
    dataset_folder = io_utils.get_dataset_folder()
    skinning_path = os.path.join(dataset_folder, 'skinning.npy')
    skeleton_path = os.path.join(dataset_folder, 'skeleton.txt')

    print('-- ANIMATION CLIPS --')
    clips = fnmatch.filter(os.listdir(dataset_folder),'*.npz')
    print(clips)
    print('num clips : ' + str(len(clips)))

    print('-- SKINNING --')
    print('skinning file : ' + str(os.path.exists(skinning_path)))

    print('-- SKELETON --')
    print('skeleton file : ' + str(os.path.exists(skeleton_path)))
    skeleton = Skeleton()
    skeleton.load(skeleton_path)
    skeleton.print_root()


if __name__ == '__main__':
    #training()
    dataset_summary()


