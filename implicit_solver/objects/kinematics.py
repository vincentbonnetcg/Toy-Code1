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
    class State:
        '''
        State of a kinematic object
        '''
        def __init__(self, position, rotation):
            self.position = position
            self.rotation = rotation # in degrees
            self.linear_velocity = np.zeros(2)
            self.angular_velocity = 0.0

        def update(self, position, rotation, dt):
            '''
            Updates the linear and angular velocity
            '''
            inv_dt = 1.0 / dt
            self.linear_velocity = np.subtract(position, self.position) * inv_dt
            shortest_angle = (rotation - self.rotation) % 360.0
            if (math.fabs(shortest_angle) > 180.0):
                shortest_angle -= 360.0
            self.angular_velocity = shortest_angle * inv_dt
            self.position = position
            self.rotation = rotation

    def __init__(self, shape):
        centroid = np.average(shape.vertex.position, axis=0)
        local_vertex_position = np.subtract(shape.vertex.position, centroid)
        self.convex_hull = ConvexHull(local_vertex_position)
        self.state = Kinematic.State(position = centroid, rotation = 0.0)
        self.index = 0 # set after the object is added to the scene - index in the scene.kinematics[]
        self.meta_data = {} # Metadata

    def set_indexing(self, index):
        self.index = index

    def get_rotation_matrix(self):
        theta = np.radians(self.state.rotation)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))

    def get_inverse_rotation_matrix(self):
        theta = np.radians(-self.state.rotation)
        c, s = np.cos(theta), np.sin(theta)
        return np.array(((c, -s), (s, c)))

    def get_vertices(self, localSpace):
        if localSpace:
            return self.convex_hull.counter_clockwise_points
        R = self.get_rotation_matrix()
        point_ws = np.matmul(self.convex_hull.counter_clockwise_points, R)
        return np.add(point_ws, self.state.position)

    def get_closest_parametric_value(self, point):
        '''
        Returns a pair [edgeId, line parameter (t)] which define
        the closest point on the convex hull
        '''
        inv_R = self.get_inverse_rotation_matrix()
        local_point = np.matmul(point - self.state.position, inv_R)
        param = self.convex_hull.get_closest_parametric_value(local_point)
        return param

    def get_point_from_parametric_value(self, param):
        local_point = self.convex_hull.get_point_from_parametric_value(param)
        R = self.get_rotation_matrix()
        return np.matmul(local_point, R) + self.state.position

    def get_normal_from_parametric_value(self, param):
        local_normal = self.convex_hull.get_normal_from_parametric_value(param)
        R = self.get_rotation_matrix()
        return np.matmul(local_normal, R)

    def is_inside(self, point):
        '''
        Returns whether or not the point is inside the kinematic
        '''
        inv_R = self.get_inverse_rotation_matrix()
        local_point = np.matmul(point - self.state.position, inv_R)
        return self.convex_hull.is_inside(local_point)
