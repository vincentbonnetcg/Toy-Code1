"""
@author: Vincent Bonnet
@description : Kinematic object describes animated objects
"""

import math
import numpy as np
from lib.common import Shape
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
        # append points
        self.point_handles =  details.point.append_empty(len(shape.vertex))
        details.point.copyto('local_x', shape.vertex, self.point_handles)
        details.point.copyto('x', shape.vertex, self.point_handles)
        point_ids = details.point.flatten('ID', self.point_handles)
        # append edges
        surface_edge_ids, surface_edge_normals = shape.get_edge_surface_data()
        edge_pids = np.take(point_ids, surface_edge_ids, axis=0)
        self.edge_handles = details.edge.append_empty(len(surface_edge_ids))
        details.edge.copyto('normal', surface_edge_normals, self.edge_handles)
        details.edge.copyto('point_IDs', edge_pids, self.edge_handles)
        # append triangles
        self.face_ids = np.copy(shape.face)
        triangle_pids = np.take(point_ids, self.face_ids, axis=0)
        self.triangle_handles = details.triangle.append_empty(len(self.face_ids))
        details.triangle.copyto('point_IDs', triangle_pids, self.triangle_handles)
        # update vertices
        self.update(details, position, rotation)
        # metadata
        self.meta_data = {}

    def get_as_shape(self, details):
        x = details.point.flatten('x', self.point_handles)
        shape = Shape(len(x), 0, len(self.face_ids))
        np.copyto(shape.vertex, x)
        np.copyto(shape.face, self.face_ids)
        return shape

    def update(self, details, position, rotation, dt = 0.0):
        # update state
        self.state.update(position, rotation, dt)
        # compute rotation matarix
        theta = np.radians(self.state.rotation)
        c, s = np.cos(theta), np.sin(theta)
        rotation_matrix = np.array(((c, -s), (s, c)))
        # update vertices
        cpn.simplex.transform_point(details.point,
                                   rotation_matrix,
                                   self.state.position,
                                   self.point_handles)
