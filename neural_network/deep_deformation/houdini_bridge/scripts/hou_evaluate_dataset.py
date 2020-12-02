"""
@author: Vincent Bonnet
@description : Python code to test the training data
"""
import numpy as np
import os
import hou

def read_dataset_from_current_frame(working_dir, sop_name, prediction=False):
    frame_id = hou.intFrame()

    # Find the name of the current animation
    anim_type = hou.parm(sop_name+'/anim_types').evalAsString()
    clip_name = hou.parm(sop_name+'/'+anim_type).evalAsString()
    clip_name = clip_name.replace('.bclip','')

    if prediction:
        clip_path = os.path.join(working_dir, 'prediction', clip_name+'.npz')
    else:
        clip_path = os.path.join(working_dir, 'dataset', clip_name+'.npz')

    if os.path.exists(clip_path):
        npzfile = np.load(clip_path)
        #bones = npzfile['bones']
        #base_skinnings = npzfile['bases']
        if prediction:
            smooth_skinnings = npzfile['predicted_smooths']
        else:
            smooth_skinnings = npzfile['smooths']
            #smooth_skinnings = npzfile['bases'] # TODO - rename smooth_skinnings

        #bone = bones[frame_id-1]
        #base_skinning = base_skinnings[frame_id-1]
        smooth_skinning = smooth_skinnings[frame_id-1]

        node = hou.pwd()
        geo = node.geometry()

        for i, point in enumerate(geo.points()):
            point.setPosition(smooth_skinning[i])
    else:
        print('the file ', clip_path, ' doesnt exist')

