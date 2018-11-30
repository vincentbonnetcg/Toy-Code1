"""
@author: Vincent Bonnet
@description : Kinematic objects used to support animated objects
"""

import math
import numpy as np
import core.shapes as shapes
from core.convex_hull import ConvexHull

class Kinematic:
    '''
    Kinematic describes the base class of a kinematic object
    '''
    def __init__(self, shape):
        centroid = np.average(shape.vertex.position, axis=0)
        local_vertex_position = np.subtract(shape.vertex.position, centroid)
        self.convex_hull = ConvexHull(local_vertex_position)
        self.position = centroid
        self.rotation = 0.0 # in degrees
        self.linear_velocity = np.zeros(2) # computed in the update function
        self.angular_velocity = 0.0 # computed in the update function
        self.last_time = 0.0 # used in the update function
        self.animationFunc = None
        self.index = 0 # set after the object is added to the scene - index in the scene.kinematics[]
        # Metadata
        self.meta_data = {}

    def set_indexing(self, index):
        self.index = index

    def get_rotation_matrix(self):
        theta = np.radians(self.rotation)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))

    def get_inverse_rotation_matrix(self):
        theta = np.radians(-self.rotation)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))

    def get_vertices(self, localSpace):
        if localSpace:
            return self.convex_hull.counter_clockwise_points
        R = self.get_rotation_matrix()
        point_ws = np.matmul(self.convex_hull.counter_clockwise_points, R)
        return np.add(point_ws, self.position)

    def update(self, time):
        if self.animationFunc:
            state = self.animationFunc(time)
            # update the linear and angular velocity
            # and position, rotation afterward.
            # it is important to keep it in this order !
            if self.last_time != time:
                inv_dt = 1.0 / (time - self.last_time)
                self.linear_velocity = np.subtract(state[0], self.position) * inv_dt
                shortest_angle = (state[1] - self.rotation) % 360.0
                if (math.fabs(shortest_angle) > 180.0):
                    shortest_angle -= 360.0

                self.angular_velocity = shortest_angle * inv_dt

            self.position = state[0]
            self.rotation = state[1]
            self.last_time = time

    def get_closest_parametric_value(self, point):
        '''
        Returns a pair [edgeId, line parameter (t)] which define
        the closest point on the polygon
        '''
        inv_R = self.get_inverse_rotation_matrix()
        local_point = np.matmul(point - self.position, inv_R)
        param = self.convex_hull.get_closest_parametric_value(local_point)
        return param

    def get_point_from_parametric_value(self, param):
        local_point = self.convex_hull.get_point_from_parametric_value(param)
        R = self.get_rotation_matrix()
        return np.matmul(local_point, R) + self.position

    def get_normal_from_parametric_value(self, param):
        local_normal = self.convex_hull.get_normal_from_parametric_value(param)
        R = self.get_rotation_matrix()
        return np.matmul(local_normal, R)

    def is_inside(self, point):
        '''
        Returns whether or not the point is inside the kinematic
        '''
        inv_R = self.get_inverse_rotation_matrix()
        local_point = np.matmul(point - self.position, inv_R)
        return self.convex_hull.is_inside(local_point)
