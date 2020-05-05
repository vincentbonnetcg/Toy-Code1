"""
@author: Vincent Bonnet
@description : Kinematic object describes animated objects
"""

import math
import numpy as np
from lib.common import Shape
import lib.common.jit.geometry_2d as geo2d_lib
import lib.objects.jit as cpn

class Kinematic:
    '''
    Kinematic describes an animated object
    '''
    class State:
        # State of a kinematic object
        def __init__(self, position, rotation):
            self.position = np.zeros(2)
            self.rotation = np.float(0.0)
            self.linear_velocity = np.zeros(2)
            self.angular_velocity = np.float(0.0)
            self.update(position, rotation)

        def update(self, position = (0.0, 0.0), rotation = 0.0, dt = 0.0):
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

    def __init__(self, details, shape, position = (0., 0.), rotation = 0.):
        self.state = Kinematic.State(position = position, rotation = rotation)
        self.local_vertex = np.copy(shape.vertex)
        # local memory - TODO remove
        self.vertex = np.copy(shape.vertex)
        self.surface_edge_ids, self.surface_edge_normals = shape.get_edge_surface_data()
        self.face_ids = np.copy(shape.face)
        # append points
        self.point_handles =  details.point.append_empty(len(self.vertex))
        details.point.copyto('x', shape.vertex, self.point_handles)
        point_ids = details.point.flatten('ID', self.point_handles)
        # append edges
        self.edge_handles = details.edge.append_empty(len(self.surface_edge_ids))
        edge_pids = np.take(point_ids, self.surface_edge_ids, axis=0)
        details.edge.copyto('normal', self.surface_edge_normals, self.edge_handles)
        details.edge.copyto('point_IDs', edge_pids, self.edge_handles)
        # append triangles
        self.triangle_handles = details.triangle.append_empty(len(self.face_ids))
        triangle_pids = np.take(point_ids, self.face_ids, axis=0)
        details.triangle.copyto('point_IDs', triangle_pids, self.triangle_handles)
        # update vertices
        self.update(details, position, rotation)
        self.index = 0 # set after the object is added to the scene - index in the scene.kinematics[]
        self.meta_data = {}

    def set_indexing(self, index):
        self.index = index

    def get_as_shape(self, details):
        shape = Shape(len(self.local_vertex), len(self.surface_edge_ids), len(self.face_ids))
        np.copyto(shape.vertex, details.point.flatten('x', self.point_handles))
        np.copyto(shape.edge, self.surface_edge_ids)
        np.copyto(shape.face, self.face_ids)
        return shape
    def update(self, details, position, rotation, dt = 0.0):
        # update state
        self.state.update(position, rotation, dt)
        # compute rotation matarix
        theta = np.radians(self.state.rotation)
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array(((c, -s), (s, c)))
        # update vertices - TODO remove
        np.dot(self.local_vertex, rotation_matrix, out=self.vertex)
        np.add(self.vertex, self.state.position, out=self.vertex)
        # update vertices
        details.point.copyto('x', self.local_vertex, self.point_handles)
        cpn.simplex.transformPoint(details.point,
                                   rotation_matrix,
                                   self.state.position,
                                   self.point_handles)

    def get_closest_param(self, details, position):
        param = geo2d_lib.ParametricPoint()
        squaredDistance = np.finfo(np.float64).max
        cpn.simplex.get_closest_param(details.edge, details.point, position,
                                      param, squaredDistance,
                                      self.edge_handles)
        return param

    def is_inside(self, point):
        return geo2d_lib.is_inside(point, self.vertex, self.face_ids)

