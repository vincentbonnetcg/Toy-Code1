"""
@author: Vincent Bonnet
@description : Python code to export bone and point data into a training folder
"""

import numpy as np
import os

# get geometry deformation
# TODO
node = hou.pwd()
geo = node.geometry()

# get bone rotation
def get_bone_rotation_y():
    b1_ry = hou.evalParm('../../chain_bone1/ry')
    b2_ry = hou.evalParm('../../chain_bone2/ry')
    b3_ry = hou.evalParm('../../chain_bone3/ry')
    b4_ry = hou.evalParm('../../chain_bone4/ry')
    array = [b1_ry, b2_ry, b3_ry, b4_ry]
    return np.array(array)

bone_rotations = get_bone_rotation_y()

# create the training directory if necessary
working_dir = os.path.dirname(hou.hipFile.path())
training_path = working_dir + '/training/' # data+labels
training_dir = os.path.dirname(training_path)
if not os.path.exists(training_dir):
    os.makedirs(training_dir)

# generate identifier for this training sample (use frameId)
traning_ID = hou.intFrame()
out_file_path = 'file' + str(traning_ID)
out_file_path =  training_path + out_file_path

# export data
print(out_file_path)
np.savez(out_file_path, rot=bone_rotations)

