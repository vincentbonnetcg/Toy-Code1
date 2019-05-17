"""
@author: Vincent Bonnet
@description :Skinning Main
"""

import hierarchy
import geometry
import render
from linear_blend_skinning import LinearBlendSkinning
from pose_space_deformer import PoseSpaceDeformer
import numpy as np

'''
User Parameters
'''
# Sequence settings
NUM_FRAMES = 97
FRAME_TIME_STEP = 1.0 / 24.0

# Geometry settings
BEAM_MIN_X = -7.0
BEAM_MIN_Y = -1.0
BEAM_MAX_X = 7.0
BEAM_MAX_Y = 1.0
BEAM_CELL_X = 20
BEAM_CELL_Y = 5

# Weight function settings
KERNEL_PARAMETER = 1.0
KERNEL_FUNCTION = lambda v : np.exp(-np.square((v * KERNEL_PARAMETER)))
BIDDING_MAX_INFLUENCES = 4

# Folder output
RENDER_FOLDER_PATH = "" # specify a folder to export png files
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif

def linear_blend_skinning():
    '''
    Linear blend skinning main
    '''
    mesh = geometry.create_beam_mesh(BEAM_MIN_X, BEAM_MIN_Y, BEAM_MAX_X, BEAM_MAX_Y, BEAM_CELL_X, BEAM_CELL_Y)
    skeleton = hierarchy.create_skeleton_with_2_bones()

    linear_blend_skinning = LinearBlendSkinning(mesh, skeleton)
    linear_blend_skinning.attach_mesh(max_influences = BIDDING_MAX_INFLUENCES, kernel_func = KERNEL_FUNCTION)
    render.draw(mesh, skeleton, 0, RENDER_FOLDER_PATH)

    for frame_id in range(1, NUM_FRAMES):
        skeleton.animate(frame_id * FRAME_TIME_STEP)
        linear_blend_skinning.update_mesh()
        render.draw(mesh, skeleton, frame_id, RENDER_FOLDER_PATH)

def pose_based_deformation():
    '''
    PSD main
    '''
    mesh = geometry.create_beam_mesh(BEAM_MIN_X, BEAM_MIN_Y, BEAM_MAX_X, BEAM_MAX_Y, BEAM_CELL_X, BEAM_CELL_Y)
    skeleton = hierarchy.create_skeleton_with_2_bones()

    # Create blendshapes from a linear blend skinning (bidding_max_influences > 1)
    linear_blend_skinning = LinearBlendSkinning(mesh, skeleton)
    bidding_max_influences = 4
    linear_blend_skinning.attach_mesh(max_influences = bidding_max_influences, kernel_func = KERNEL_FUNCTION)
    render.draw(mesh, skeleton, 0, RENDER_FOLDER_PATH)


def main():
    '''
    Main
    '''
    pose_based_deformation();

if __name__ == '__main__':
    main()

