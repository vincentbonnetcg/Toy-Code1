"""
@author: Vincent Bonnet
@description : Convex Hull wrapper
"""

import numpy as np
import scipy.spatial as spatial
import core.math_2d as math2D

class ConvexHull:

    def __init__(self, points):
        hull = spatial.ConvexHull(points)
        self.points = np.copy(points)
        self.simplices = hull.simplices
        self.equations = hull.equations
        self.counter_clockwise_points = np.take(hull.points, hull.vertices, axis=0)

    def get_closest_parametric_value(self, point):
        '''
        Projects the point on the closest segment of the convex hull and
        returns the closest parametric value as a tuple [edgeId, ratio]
        self.equations is an array of equation [a, b, c]
        '''
        result = [-1, 0.0] # [simplices index, parameteric value]

        # Search for the closest segment
        min_distance2 = np.finfo(np.float64).max
        for index, equation in enumerate(self.equations):
            plane_distance = get_closest_distance(equation, point)
            # projects point on the current plane
            plane_normal = equation[0:2]
            projected_point = point + plane_distance * plane_normal
            # compute the distance from the segment
            segment_pts = np.take(self.points, self.simplices[index], axis=0)# could be precomputed
            segment_direction = segment_pts[1] - segment_pts[0] # could be precomputed
            segment_direction_dot = math2D.dot(segment_direction, segment_direction)     # could be precomputed
            direction = projected_point - segment_pts[0]

            t = math2D.dot(direction, segment_direction) / segment_direction_dot
            t = max(min(t, 1.0), 0.0)
            projected_point = segment_pts[0] + segment_direction * t # correct the project point
            vector_distance = (point - projected_point)
            distance2 = math2D.dot(vector_distance, vector_distance)
            # update the minimum distance
            if distance2 < min_distance2:
                result = [index, t]
                min_distance2 = distance2

        return result

    def get_point_from_parametric_value(self, param):
        index, t = param
        segment_pts = np.take(self.points, self.simplices[index], axis=0)# could be precomputed
        segment_direction = segment_pts[1] - segment_pts[0] # could be precomputed
        return segment_pts[0] + segment_direction * t

    def get_normal_from_parametric_value(self, param):
        index, t = param
        equation = self.equations[index]
        return equation[0:2]

    def is_inside(self, point):
        '''
        Returns whether or not the point is inside the kinematic
        The base is not implemented yet
        '''
        inside = True
        for equation in self.equations:
            if get_closest_distance(equation, point) < 0.0:
                inside = False

        return inside

def get_closest_distance(equation, point):
    '''
    Closest distance from the 2D-plane equation
    2D-plane equation is defined by : ax + by + c = 0
    where a, b, c is the plane equation
    We are looking for
    x = x0 + a*d and y = y0 + b*d
    by substitution
    a*(x0 + a*d) + b*(y0 + b*d) + c = 0
    hence
    d = - (a*x+b*y+c) / (a**2 + b**2)
    '''
    a, b, c = equation
    x, y = point
    d = - (a*x+b*y+c) / (a**2 + b**2)
    return d
