"""
@author: Vincent Bonnet
@description : Python code to export bone and point data into a training folder
"""
import numpy as np
import os

# get bone rotation
def get_bone_rotation_y():
    b1_ry = hou.evalParm('../../chain_bone1/ry')
    b2_ry = hou.evalParm('../../chain_bone2/ry')
    b3_ry = hou.evalParm('../../chain_bone3/ry')
    b4_ry = hou.evalParm('../../chain_bone4/ry')
    array = [b1_ry, b2_ry, b3_ry, b4_ry]
    return np.array(array)

bone_rotations = get_bone_rotation_y()

# get undeformed and deformed point data
def get_point_data(input_id):
    node = hou.pwd()
    inputs = node.inputs()
    if input_id >= len(inputs):
        raise Exception('input_id >= len(inputs)')

    input_geo = inputs[input_id].geometry()
    points = input_geo.points()
    num_vertices = len(points)
    pos_array = np.zeros((num_vertices, 3), dtype=float)
    for i, point in enumerate(points):
        point = point.position()
        pos_array[i] = [point[0],point[1],point[2]]

    return pos_array

deformed_point_data = get_point_data(0)
undeformed_point_data = get_point_data(1)
deformed_offset_data = deformed_point_data - undeformed_point_data

# create the training directory if necessary
working_dir = os.path.dirname(hou.hipFile.path())
training_path = working_dir + '/training/' # data+labels
training_dir = os.path.dirname(training_path)
if not os.path.exists(training_dir):
    os.makedirs(training_dir)

# generate identifier for the current training sample (use frameId)
traning_ID = hou.intFrame()
out_file_path = 'file' + str(traning_ID)
out_file_path =  training_path + out_file_path

# export data
np.savez(out_file_path, rot=bone_rotations, undeformed=deformed_point_data, offset=deformed_offset_data)

