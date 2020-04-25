"""
@author: Vincent Bonnet
@description : Python code to test the training data
"""
import numpy as np
import os
import hou

def read_dataset_from_current_frame(working_dir):
    dataset_dir = os.path.join(working_dir, 'dataset')
    data_ID = hou.intFrame()
    file_path = 'file' + str(data_ID) + '.npz'
    file_path =  os.path.join(dataset_dir, file_path)

    # Update position from file
    if os.path.exists(file_path):
        npzfile = np.load(file_path)

        bone_infos = npzfile['bone_infos']
        rigid_skinning = npzfile['rigid_skinning']
        smooth_skinning = npzfile['smooth_skinning']

        node = hou.pwd()
        geo = node.geometry()
        for i, point in enumerate(geo.points()):
            point.setPosition(smooth_skinning[i])
    else:
        print('the file ', file_path, ' doesnt exist')
