"""
@author: Vincent Bonnet
@description : Python code to export bone and point data into a dataset folder
"""
import numpy as np
import os
import hou

BASE_SKINNING_INPUT_ID = 0
SMOOTH_SKINNING_INPUT_ID = 1
DATASET_FOLDER = 'dataset'


def get_bone_data(sop_name, bone_names):
    # sop_name : name of the sop network containing the bones
    # bone_names : name of the node representing the bone
    attributes = ['rx', 'ry', 'rz', 'length']
    result = np.empty((len(bone_names), len(attributes)), dtype=float)
    for i, bone_name in enumerate(bone_names):
        for j, attr in enumerate(attributes):
            result[i,j] = hou.evalParm(sop_name+bone_name+'/'+attr)

    return result

def get_bone_parents(sop_name, bone_names):
    parents = []
    for bone_name in bone_names:
        node = hou.node(sop_name+bone_name)
        parentName = node.inputs()[0].name()
        parents.append(parentName)

    return parents

def get_geo(input_id):
    node = hou.pwd()
    inputs = node.inputs()
    if input_id >= len(inputs):
        raise Exception('input_id >= len(inputs)')
    return inputs[input_id].geometry()

def get_vertices(input_id):
    points = get_geo(input_id).points()
    num_points = len(points)

    pos_array = np.zeros((num_points, 3), dtype=float)
    for i, point in enumerate(points):
        pt = point.position()
        pos_array[i] = [pt[0],pt[1],pt[2]]

    return pos_array

def get_bone_names(input_id):
    geo = get_geo(input_id)
    regions = geo.stringListAttribValue('boneCapture_pCaptPath')
    bone_names = []
    for region in regions:
        if '/cregion 0' in region:
            bone_names.append(region.replace('/cregion 0', ''))
        else:
            raise Exception('region format not supported')
    return bone_names

def get_skinning_data(input_id):
    geo = get_geo(input_id)
    points = geo.points()
    num_points = len(points)

    # get max influence per attribute
    max_influences = 0
    for point in points:
        boneids = point.intListAttribValue('boneCapture_index')
        if len(boneids) > max_influences:
            max_influences = len(boneids)

    # extract skinning data
    data_type = {}
    data_type['names'] = ['numInfluences', 'boneIds', 'weights']
    data_type['formats'] = ['int8', ('int8', max_influences), ('float32', max_influences)]
    skinning_data = np.zeros(num_points, dtype=np.dtype(data_type, align=True))

    for i, point in enumerate(points):
        boneIds = point.intListAttribValue('boneCapture_index')
        weights = point.floatListAttribValue('boneCapture_data')
        num_influences = len(boneIds)
        skinning_data[i]['numInfluences'] = num_influences
        for j in range(num_influences):
            skinning_data[i]['boneIds'][j] = boneIds[j]
            skinning_data[i]['weights'][j] = weights[j]

    return skinning_data


def prepare_dataset_dir(working_dir):
    dataset_dir = os.path.join(working_dir, DATASET_FOLDER)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    return dataset_dir

def export_data_from_current_frame(working_dir, sop_name):
    max_frames = hou.evalParm(sop_name+'/nFrames')
    frame_id = hou.intFrame()
    if frame_id > max_frames or frame_id <= 0:
        print('do not write frame_id({}) because > max_frames({})'.format(frame_id, max_frames))
        return

    # Find the name of the current animation
    anim_type = hou.parm(sop_name+'/anim_types').evalAsString()
    clip_name = hou.parm(sop_name+'/'+anim_type).evalAsString()
    clip_name = clip_name.replace('.bclip','')

    # Export skeleton
    # The skeleton hierarchy is frame invariant => only write it once
    bone_names = get_bone_names(SMOOTH_SKINNING_INPUT_ID)
    skeleton_path = os.path.join(working_dir, DATASET_FOLDER, 'skeleton.txt')
    if not os.path.exists(skeleton_path):
        parent_names = get_bone_parents(sop_name, bone_names)
        with open(skeleton_path, 'w') as file_handler:
            for i, bone_name in enumerate(bone_names):
                file_handler.write(bone_name + ',' + parent_names[i])
                file_handler.write('\n')

    # Export skinning
    # The skinning data is frame invariant => only write it once
    skinning_data = get_skinning_data(SMOOTH_SKINNING_INPUT_ID)
    skinning_path = os.path.join(working_dir, DATASET_FOLDER, 'skinning.npy')
    if not os.path.exists(skinning_path):
        np.save(skinning_path, skinning_data)

    # Export bone and geometry dataset
    bone = get_bone_data(sop_name, bone_names)
    base_skinning = get_vertices(BASE_SKINNING_INPUT_ID)
    smooth_skinning = get_vertices(SMOOTH_SKINNING_INPUT_ID)

    clip_path = os.path.join(working_dir, DATASET_FOLDER, clip_name+'.npz')
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
