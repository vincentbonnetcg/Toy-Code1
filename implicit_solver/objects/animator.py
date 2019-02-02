"""
@author: Vincent Bonnet
@description : Animator stores the position and rotation per frame
"""
import numpy as np

class Animator:
    '''
    Animator class to store the position and rotation per frame
    '''
    def __init__(self, lambda_func, context):
        self.num_frames = context.num_frames
        self.start_time = context.start_time
        self.end_time = context.end_time
        self.frame_dt = context.frame_dt
        self.positions = np.zeros((self.num_frames, 2), dtype=float)
        self.rotations = np.zeros((self.num_frames, 2), dtype=float)
        for frame_id in range(self.num_frames):
            time = self.start_time + (frame_id * self.frame_dt)
            position, rotation = lambda_func(time)
            self.positions[frame_id] = position
            self.rotations[frame_id] = rotation

    def get_value(self, time):
        pass
