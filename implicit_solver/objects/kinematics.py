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
        self.meta_data = {} # Metadata

    def set_indexing(self, index):
        self.index = index

    def get_vertices(self, localSpace):
        if localSpace:
            return self.convex_hull.counter_clockwise_points
        theta = np.radians(self.rotation)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        result = np.matmul(self.convex_hull.counter_clockwise_points.copy(), R)
        result = np.add(result, self.position)
        return result

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

    def getClosestParametricValues(self, point):
         # TODO : CALL CONVEX_HULL
        '''
        Returns a pair [edgeId, line parameter (t)] which define
        the closest point on the polygon
        '''
        worldSpaceVertices = self.get_vertices(False)
        if len(worldSpaceVertices) == 0:
            return None
        elif len(worldSpaceVertices) == 1:
            return [-1, 0.0] # No edge

        result = [-1, 0.0]
        minDistance = np.finfo(np.float64).max
        numEdges = len(worldSpaceVertices)
        for edgeId in range(numEdges):
            # project point on line
            edge = worldSpaceVertices[(edgeId+1)%numEdges] - worldSpaceVertices[edgeId]
            d = point - worldSpaceVertices[edgeId]
            u2 = np.inner(edge, edge)
            t = np.dot(d, edge) / u2
            t = max(min(t, 1.0), 0.0)
            projectedPoint = worldSpaceVertices[edgeId] + (t * edge)
            normal = (point - projectedPoint)
            dist2 = np.inner(normal, normal)
            if dist2 < minDistance:
                minDistance = dist2
                result = [edgeId, t]

        return result

    def getPointFromParametricValues(self, parametricValues):
         # TODO : CALL CONVEX_HULL
        worldSpaceVertices = self.get_vertices(False)
        numEdges = len(worldSpaceVertices)
        edgeId = parametricValues[0]
        if edgeId == -1: # a single point
            return worldSpaceVertices[0]
        else:
            A = worldSpaceVertices[edgeId]
            B = worldSpaceVertices[(edgeId+1)%numEdges]
            t = parametricValues[1]
            return A * (1.0 - t) + B * t

        return None

    def getNormalFromParametricValues(self, parametricValues):
         # TODO : CALL CONVEX_HULL
        worldSpaceVertices = self.get_vertices(False)
        numEdges = len(worldSpaceVertices)
        edgeId = parametricValues[0]
        if edgeId >= 0:

            # Compute the normal of the convex hull in local space
            A = self.convex_hull.counter_clockwise_points[edgeId]
            B = self.convex_hull.counter_clockwise_points[(edgeId+1)%numEdges]
            t = parametricValues[1]
            p = A * (1.0 - t) + B * t
            n = [A[1] - B[1], A[0] - B[0]]
            n /= np.linalg.norm(n)

            dot = n[0] * p[0] + n[1] * p[1]
            if (dot < 0.0):
                n[0] *= -1.0
                n[1] *= -1.0

            # Transform the local space normal to world space normal
            theta = np.radians(self.rotation)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            n = np.matmul(n, R)

            return n

        return [0.0, 0.0]

    def is_inside(self, point):
        '''
        Returns whether or not the point is inside the kinematic
        '''
        theta = np.radians(-self.rotation)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        local_point = np.matmul(point - self.position, R)
        return self.convex_hull.is_inside(local_point)
