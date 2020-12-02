"""
@author: Vincent Bonnet
@description : Python code to export bone and point data into a dataset folder
"""
import numpy as np
import os
import hou

BASE_SKINNING_INPUT_ID = 0
SMOOTH_SKINNING_INPUT_ID = 1

def get_bone_infos(sop_name, bone_names):
    # sop_name : name of the sop network containing the bones
    # bone_names : name of the node representing the bone
    attributes = ['rx', 'ry', 'rz', 'length']
    result = np.empty((len(bone_names), len(attributes)), dtype=float)
    for i, bone_name in enumerate(bone_names):
        for j, attr in enumerate(attributes):
            result[i,j] = hou.evalParm(sop_name+bone_name+'/'+attr)

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

def export_data_from_current_frame(working_dir, sop_name, bone_names):
    max_frames = hou.evalParm(sop_name+'/nFrames')
    frame_id = hou.intFrame()
    if frame_id > max_frames or frame_id <= 0:
        print('do not write frame_id({}) because > max_frames({})'.format(frame_id, max_frames))
        return

    # Find the name of the current animation
    anim_type = hou.parm(sop_name+'/anim_types').evalAsString()
    clip_name = hou.parm(sop_name+'/'+anim_type).evalAsString()
    clip_name = clip_name.replace('.bclip','')

    # Get data
    bone = get_bone_infos(sop_name, bone_names)
    base_skinning = get_points(BASE_SKINNING_INPUT_ID)
    smooth_skinning = get_points(SMOOTH_SKINNING_INPUT_ID)

    clip_path = os.path.join(working_dir, 'dataset', clip_name+'.npz')

    bones, base_skinnings, smooth_skinnings = None, None, None
    if not os.path.exists(clip_path):
        bone_shape = ([max_frames] + list(bone.shape))
        base_shape = ([max_frames] + list(base_skinning.shape))
        smooth_shape = ([max_frames] + list(smooth_skinning.shape))

        bones = np.empty(bone_shape, dtype=bone.dtype)
        base_skinnings = np.empty(base_shape, dtype=base_skinning.dtype)
        smooth_skinnings = np.empty(smooth_shape, dtype=smooth_skinning.dtype)
    else:
        npzfile = np.load(clip_path)
        bones = npzfile['bones']
        base_skinnings = npzfile['bases']
        smooth_skinnings = npzfile['smooths']

    # Save data
    bones[frame_id-1] = bone
    base_skinnings[frame_id-1] = base_skinning
    smooth_skinnings[frame_id-1] = smooth_skinning

    out_attributes = {'bones' : bones,
                      'bases' : base_skinnings,
                      'smooths' : smooth_skinnings}
    np.savez(clip_path, **out_attributes)

    print('writing frame {}/{} from animation into the file : {}'.format(frame_id, max_frames, clip_path))

