"""
@author: Vincent Bonnet
@description : Kinematic object describes animated objects
"""

import math
import numpy as np
from lib.common.convex_hull import ConvexHull
from lib.common import Shape

class Kinematic:
    '''
    Kinematic describes an animated object
    '''
    class State:
        '''
        State of a kinematic object
        '''
        def __init__(self, position, rotation):
            self.position = np.zeros(2)
            self.rotation = np.float(0.0)
            self.linear_velocity = np.zeros(2)
            self.angular_velocity = np.float(0.0)
            self.rotation_matrix = np.zeros((2,2))
            self.inverse_rotation_matrix = np.zeros((2,2))
            self.update(position, rotation)

        def update(self, position = (0.0, 0.0), rotation = 0.0, dt = 0.0):
            '''
            Updates the state
            '''
            # Updates linear and angular velocity
            if dt > 0.0:
                inv_dt = 1.0 / dt
                self.linear_velocity = np.subtract(position, self.position) * inv_dt
                shortest_angle = (rotation - self.rotation) % 360.0
                if (math.fabs(shortest_angle) > 180.0):
                    shortest_angle -= 360.0
                    self.angular_velocity = shortest_angle * inv_dt

            # Updates position and rotation
            self.position = np.asarray(position)
            self.rotation = np.float(rotation)

            # Update rotation matrices
            theta = np.radians(self.rotation)
            c, s = np.cos(theta), np.sin(theta)
            self.rotation_matrix = np.array(((c, -s), (s, c)))

            c, s = np.cos(-theta), np.sin(-theta)
            self.inverse_rotation_matrix = np.array(((c, -s), (s, c)))

    def __init__(self, shape, position = (0., 0.), rotation = 0.):
        self.convex_hull = ConvexHull(shape.vertex)
        self.state = Kinematic.State(position = position, rotation = rotation)
        self.vertices = np.copy(shape.vertex)
        self.edge_ids = np.copy(shape.edge)
        self.face_ids = np.copy(shape.face)
        self.index = 0 # set after the object is added to the scene - index in the scene.kinematics[]
        self.meta_data = {} # Metadata

    def set_indexing(self, index):
        self.index = index

    def get_shape(self):
        shape = Shape(len(self.vertices), len(self.edge_ids), len(self.face_ids))
        np.copyto(shape.vertex, self.vertices)
        np.copyto(shape.edge, self.edge_ids)
        np.copyto(shape.face, self.face_ids)

        np.matmul(shape.vertex, self.state.rotation_matrix, out=shape.vertex)
        np.add(shape.vertex, self.state.position, out=shape.vertex)

        return shape

    def get_closest_parametric_point(self, point):
        '''
        Returns a pair [edgeId, line parameter (t)] which define
        the closest point on the convex hull
        '''
        inv_R = self.state.inverse_rotation_matrix
        local_point = np.matmul(point - self.state.position, inv_R)
        param = self.convex_hull.get_closest_parametric_point(local_point)
        return param

    def get_position_from_parametric_point(self, param):
        local_point = self.convex_hull.get_position_from_parametric_point(param)
        R = self.state.rotation_matrix
        return np.matmul(local_point, R) + self.state.position

    def get_normal_from_parametric_point(self, param):
        local_normal = self.convex_hull.get_normal_from_parametric_point(param)
        R = self.state.rotation_matrix
        return np.matmul(local_normal, R)

    def is_inside(self, point):
        '''
        Returns whether or not the point is inside the kinematic
        '''
        inv_R = self.state.inverse_rotation_matrix
        local_point = np.matmul(point - self.state.position, inv_R)
        return self.convex_hull.is_inside(local_point)
