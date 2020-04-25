"""
@author: Vincent Bonnet
@description : Python code to export bone and point data into a dataset folder
"""
import numpy as np
import os
import hou

RIGID_SKINNING_INPUT_ID = 0
SMOOTH_SKINNING_INPUT_ID = 1

def get_bone_infos():
    bones = ['chain_bone1', 'chain_bone2', 'chain_bone3', 'chain_bone4']
    attributes = ['rx', 'ry', 'rz', 'length']
    result = np.empty((len(bones), len(attributes)), dtype=float)
    for i, bone in enumerate(bones):
        for j, attr in enumerate(attributes):
            result[i,j] = hou.evalParm('../../'+bone+'/'+attr)

    return result

def get_points(input_id):
    node = hou.pwd()
    inputs = node.inputs()
    if input_id >= len(inputs):
        raise Exception('input_id >= len(inputs)')

    points = inputs[input_id].geometry().points()
    num_vertices = len(points)
    pos_array = np.zeros((num_vertices, 3), dtype=float)
    for i, point in enumerate(points):
        point = point.position()
        pos_array[i] = [point[0],point[1],point[2]]

    return pos_array

def prepare_dataset_dir(working_dir):
    dataset_dir = os.path.join(working_dir, 'dataset')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    return dataset_dir

def export_data_from_current_frame(working_dir):
    dataset_dir = os.path.join(working_dir, 'dataset')
    data_ID = hou.intFrame()
    out_file_path = 'file' + str(data_ID)
    out_file_path =  os.path.join(dataset_dir, out_file_path)

    # Get data
    rigid_skinning = get_points(RIGID_SKINNING_INPUT_ID)
    smooth_skinning = get_points(SMOOTH_SKINNING_INPUT_ID)
    bone_infos = get_bone_infos()

    # Export data
    output_attributes = {}
    output_attributes['bone_infos'] = bone_infos
    output_attributes['rigid_skinning'] = rigid_skinning
    output_attributes['smooth_skinning'] = smooth_skinning

    np.savez(out_file_path, **output_attributes)
