"""
@author: Vincent Bonnet
@description : Animator stores the position and rotation per frame
"""
import numpy as np
import math
import lib.objects.jit.algorithms.simplex_lib as simplex_lib
import core.jit.math_2d as math2D

class Animator:
    '''
    Animator class to store the position and rotation per frame
    '''
    def __init__(self, lambda_func, context):
        # animation data
        num_baked_frames = context.num_frames + 1
        self.num_frames = context.num_frames # simulated frame
        self.frame_dt = context.frame_dt
        self.positions = np.zeros((num_baked_frames, 2), dtype=float)
        self.rotations = np.zeros(num_baked_frames, dtype=float)
        self.times = np.linspace(context.start_time, context.end_time,
                                 num=num_baked_frames, dtype=float)
        for frame_id in range(num_baked_frames):
            time = self.times[frame_id]
            position, rotation = lambda_func(time)
            self.positions[frame_id] = position
            self.rotations[frame_id] = rotation

        # state
        self.position = np.zeros(2)
        self.rotation = np.float(0.0)
        self.linear_velocity = np.zeros(2)
        self.angular_velocity = np.float(0.0)
        self.update_state(position, rotation)

    def get_value(self, time):
        start_time = self.times[0]
        end_time = self.times[-1]
        # Compute the frame ids contributing to the current time
        # Instead of implementing a search on self.times, the frame_dt is used
        relative_frame = (time - start_time) * self.num_frames
        relative_frame /= (end_time - start_time)
        relative_frame = min(self.num_frames, max(0, relative_frame))
        frame_ids = (math.floor(relative_frame), math.ceil(relative_frame))

        # Compute the animated values (position / rotation)
        # Special case when landing on a baked frame
        if (frame_ids[0] == frame_ids[1]):
            i = frame_ids[0]
            return (self.positions[i], self.rotations[i])

        # Linear interpolation of the values (position / rotation)
        times = (self.times[frame_ids[0]], self.times[frame_ids[1]])
        assert(time >= times[0] and time <= times[1])
        weight = (time - times[0]) / (times[1] - times[0])

        position = self.positions[frame_ids[1]] * weight
        position += self.positions[frame_ids[0]] * (1.0 - weight)
        rotation = self.rotations[frame_ids[1]] * weight
        rotation += self.rotations[frame_ids[0]] * (1.0 - weight)

        return (position, rotation)

    def update_state(self, position, rotation, dt = 0.0):
        # Updates linear and angular velocity
        if dt > 0.0:
            inv_dt = 1.0 / dt
            self.linear_velocity = np.subtract(position, self.position) * inv_dt
            shortest_angle = (rotation - self.rotation) % 360.0
            if (math.fabs(shortest_angle) > 180.0):
                shortest_angle -= 360.0
                self.angular_velocity = shortest_angle * inv_dt

        # update position and rotation
        self.position = np.asarray(position)
        self.rotation = np.float(rotation)

    def update_kinematic(self, details, kinematic, context):
        # update state
        position, rotation = self.get_value(context.time)
        self.update_state(position, rotation, context.dt)
        # update kinematic
        rotation_matrix =  math2D.rotation_matrix(rotation)
        simplex_lib.transform_point(details.point,
                                rotation_matrix,
                                self.position,
                                kinematic.point_handles)

        simplex_lib.transform_normal(details.edge,
                                 rotation_matrix,
                                 kinematic.edge_handles)

