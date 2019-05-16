"""
@author: Vincent Bonnet
@description :Skinning Main
"""

import hierarchy
import geometry
import render
import numpy as np

'''
User Parameters
'''
NUM_FRAMES = 97
FRAME_TIME_STEP = 1.0 / 24.0

BEAM_MIN_X = -7.0
BEAM_MIN_Y = -1.0
BEAM_MAX_X = 7.0
BEAM_MAX_Y = 1.0
BEAM_CELL_X = 20
BEAM_CELL_Y = 5

BIDDING_MAX_INFLUENCES = 4

RENDER_FOLDER_PATH = "" # specify a folder to export png files
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif

def main():
    '''
    Main
    '''
    mesh = geometry.create_beam_mesh(BEAM_MIN_X, BEAM_MIN_Y, BEAM_MAX_X, BEAM_MAX_Y, BEAM_CELL_X, BEAM_CELL_Y)
    skeleton = hierarchy.create_skeleton_with_2_bones()

    kernel_parameter = 1.0
    kernel_function = lambda v : np.exp(-np.square((v * kernel_parameter)))
    skeleton.attach_mesh(mesh, max_influences = BIDDING_MAX_INFLUENCES, kernel_func = kernel_function)
    render.draw(mesh, skeleton, 0, RENDER_FOLDER_PATH)

    for frame_id in range(1, NUM_FRAMES):
        skeleton.animate(frame_id * FRAME_TIME_STEP)
        skeleton.update_mesh(mesh)
        render.draw(mesh, skeleton, frame_id, RENDER_FOLDER_PATH)

if __name__ == '__main__':
    main()

