"""
@author: Vincent Bonnet
@description :Skinning Main
"""

import hierarchy
import geometry
import render
from linear_blend_skinning import LinearBlendSkinning
from pose_space_deformer import PoseSpaceDeformer
import pose_space_deformer
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
BEAM_CELL_Y = 2

# Weight function settings
KERNEL_PARAMETER = 1.0
KERNEL_FUNCTION = lambda v : np.exp(-np.square((v * KERNEL_PARAMETER)))
BIDDING_MAX_INFLUENCES = 2

# Pose Space Deformation
NUM_POSES = 10

# Folder output
RENDER_FOLDER_PATH = "" # specify a folder to export png files
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif

def linear_blend_skinning():
    '''
    Linear blend skinning main (or Smooth skinning)
    '''
    mesh = geometry.create_beam_mesh(BEAM_MIN_X, BEAM_MIN_Y, BEAM_MAX_X, BEAM_MAX_Y, BEAM_CELL_X, BEAM_CELL_Y)
    skeleton = hierarchy.create_skeleton_with_2_bones()

    linear_blend_skinning = LinearBlendSkinning(mesh, skeleton)
    linear_blend_skinning.attach_mesh(max_influences = BIDDING_MAX_INFLUENCES, kernel_func = KERNEL_FUNCTION)

    for frame_id in range(NUM_FRAMES):
        skeleton.animate(frame_id * FRAME_TIME_STEP)
        linear_blend_skinning.update_mesh()
        render.draw(mesh, skeleton, linear_blend_skinning.weights_map, None, frame_id, RENDER_FOLDER_PATH)

def pose_based_deformation():
    '''
    PSD main
    '''
    smooth_mesh = geometry.create_beam_mesh(BEAM_MIN_X, BEAM_MIN_Y, BEAM_MAX_X, BEAM_MAX_Y, BEAM_CELL_X, BEAM_CELL_Y)
    rigid_mesh = geometry.create_beam_mesh(BEAM_MIN_X, BEAM_MIN_Y, BEAM_MAX_X, BEAM_MAX_Y, BEAM_CELL_X, BEAM_CELL_Y)
    skeleton = hierarchy.create_skeleton_with_2_bones()
    pose_deformers = PoseSpaceDeformer()

    # Training Part
    # Create poses from a SmoothSkinning (max_influences > 1) and RigidSkinning (max_influences = 1)
    smooth_skinning = LinearBlendSkinning(smooth_mesh, skeleton)
    smooth_skinning.attach_mesh(max_influences = 4, kernel_func = KERNEL_FUNCTION)
    rigid_skinning = LinearBlendSkinning(rigid_mesh, skeleton)
    rigid_skinning.attach_mesh(max_influences = 1, kernel_func = KERNEL_FUNCTION)

    for pose_id in range(NUM_POSES):
        # create a new pose
        frame_id = pose_id * (NUM_FRAMES / NUM_POSES)
        skeleton.animate(frame_id * FRAME_TIME_STEP)
        smooth_skinning.update_mesh()
        rigid_skinning.update_mesh()

        # record the new feature and mesh from the pose
        feature = pose_space_deformer.feature_from_skeleton(skeleton)
        psd_target = smooth_skinning.mesh.vertex_buffer
        underlying_surface = rigid_skinning.mesh.vertex_buffer

        pose_deformers.add_pose(feature, underlying_surface, psd_target)
        last_displacement = pose_deformers.displacements[-1]
        render.draw(rigid_mesh, skeleton, smooth_skinning.weights_map, last_displacement, frame_id, RENDER_FOLDER_PATH)

def main():
    '''
    Main
    '''
    pose_based_deformation();

if __name__ == '__main__':
    main()

