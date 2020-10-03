"""
@author: Vincent Bonnet
@description : Deep deformation
"""

# A. add normalize inputs
# B. load_dataset() should also return validation datasets
# C. load_dataset() should use multiple datasets

import numpy as np
import deform_utils

def main():
    # Get Data
    dataset_file = 'SideFX_Male_walk_startle_L_L_001.npz'
    x_train, y_train, x_test, y_test, train_ids, test_ids = deform_utils.load_dataset(dataset_file)
    in_shape = x_train.shape[1]
    out_shape = y_train.shape[1]

    # Get model
    model = deform_utils.get_model(in_shape, out_shape)

    # Train data
    model.fit(x=x_train, y=y_train,
              epochs=100,
              batch_size=10,
              shuffle=True,
              validation_data=(x_test, y_test)) # TODO - need validation

    # Predict from test
    predicted = model.predict(x_test)
    print('---- errors ---- ')
    error = np.sqrt((y_test-predicted)**2)
    print(np.min(error), np.max(error), np.average(error))
    deform_utils.write_predicted_dataset(dataset_file, predicted, test_ids)

    # Predicted from train
    deform_utils.write_predicted_dataset(dataset_file, model.predict(x_train), train_ids)

if __name__ == '__main__':
    main()
