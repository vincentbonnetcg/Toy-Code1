"""
@author: Vincent Bonnet
@description : Animator stores the position and rotation per frame
"""
import numpy as np
import math

class Animator:
    '''
    Animator class to store the position and rotation per frame
    '''
    def __init__(self, lambda_func, context):
        self.num_frames = context.num_frames # simulated frame
        self.num_baked_frames = self.num_frames + 1 # include initial frame
        self.start_time = context.start_time
        self.end_time = context.end_time
        self.frame_dt = context.frame_dt
        self.positions = np.zeros((self.num_baked_frames, 2), dtype=float)
        self.rotations = np.zeros(self.num_baked_frames, dtype=float)
        self.times = np.zeros(self.num_baked_frames, dtype=float)
        for frame_id in range(self.num_baked_frames):
            time = self.start_time + (frame_id * self.frame_dt)
            position, rotation = lambda_func(time)
            self.positions[frame_id] = position
            self.rotations[frame_id] = rotation
            self.times[frame_id] = time

    def get_value(self, time):
        # Compute the frame ids contributing to the current time
        # Instead of implementing a search on self.times, the frame_dt is used
        relative_frame = (time - self.start_time) * self.num_frames
        relative_frame /= (self.end_time - self.start_time)
        relative_frame = min(self.num_frames, max(0, relative_frame))
        frame_ids = (math.floor(relative_frame), math.ceil(relative_frame))

        # Compute the animated values (position / rotation)
        # Special case when landing on a baked frame
        if (frame_ids[0] == frame_ids[1]):
            i = frame_ids[0]
            return (self.positions[i], self.rotations[i])

        # Linear interpolation of the values (position / rotation)
        frame_times = (self.start_time + (frame_ids[0] * self.frame_dt),
                       self.start_time + (frame_ids[1] * self.frame_dt))

        assert(time >= frame_times[0] and time <= frame_times[1])
        weight = (time - frame_times[0]) / (frame_times[1] - frame_times[0])

        position = self.positions[frame_ids[1]] * weight
        position += self.positions[frame_ids[0]] * (1.0 - weight)
        rotation = self.rotations[frame_ids[1]] * weight
        rotation += self.rotations[frame_ids[0]] * (1.0 - weight)

        return (position, rotation)
