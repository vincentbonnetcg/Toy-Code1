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
        self.simplices = hull.simplices
        self.equations = hull.equations
        self.counter_clockwise_points = np.take(hull.points, hull.vertices, axis=0)

    def get_projection_parameter(self, equation, point):
        '''
        2D-plane equation is defined by : ax + by + c = 0
        where a, b, c = equation
        We are looking for
        x = x0 + a*t and y = y0 + b*t
        by substitution
        a*(x0 + a*t) + b*(y0 + b*t) + c = 0
        hence
        t = - (a*x+b*y+c) / (a**2 + b**2)
        '''
        a, b, c = equation
        x, y = point
        t = - (a*x+b*y+c) / (a**2 + b**2)
        return t

    def get_closest_param(self, point):
        '''
        Returns the closest parametric value as a tuple [edgeId, ratio]
        self.equations is an array of equation [a, b, c]
        '''
        # TODO
        pass

    def get_point_from_param(self, param):
        # TODO
        pass

    def get_normal_from_param(self, param):
        # TODO
        pass

    def is_inside(self, point):
        '''
        Returns whether or not the point is inside the kinematic
        The base is not implemented yet
        '''
        inside = True
        for equation in self.equations:
            if self.get_projection_parameter(equation, point) < 0.0:
                inside = False

        return inside