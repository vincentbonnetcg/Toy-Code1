"""
@author: Vincent Bonnet
@description : Python code to test the training data (only for debugging)
"""
import numpy as np
import os

# get the training directory
working_dir = os.path.dirname(hou.hipFile.path())
training_path = working_dir + '/training/' # data+labels

# training identifier
traning_ID = hou.intFrame()
file_path = 'file' + str(traning_ID) + '.npz'
file_path =  training_path + file_path

# update position from file
if os.path.exists(file_path):
    npzfile = np.load(file_path)
    deformed_offset_data = npzfile['offset']
    undeformed_point_data = npzfile['undeformed']
    bone_rotations = npzfile['rot']
    deformed_point_data = undeformed_point_data + deformed_offset_data

    node = hou.pwd()
    geo = node.geometry()
    for id, point in enumerate(geo.points()):
        point.setPosition(deformed_point_data[id])

