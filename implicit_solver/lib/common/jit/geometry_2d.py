"""
@author: Vincent Bonnet
@description : Geometry methods
"""

import numba
import numpy as np
import lib.common.jit.data_accessor as db

closestResultSpec = [('points', numba.int32[:,:]), # two points
                  ('t', numba.float32), # parametric value
                  ('position', numba.float64[:]),  # position
                  ('normal', numba.float64[:]),# normal
                  ('squared_distance', numba.float64)] # parametric value
@numba.experimental.jitclass(closestResultSpec)
class ClosestResult(object):
    def __init__(self):
        self.points = db.empty_data_ids(2)
        self.t = 0.0
        self.position = np.zeros(2, dtype=np.float64)
        self.normal = np.zeros(2, dtype=np.float64)
        self.squared_distance = np.finfo(np.float64).max

insideResultSpec = [('isInside', numba.boolean)]
@numba.experimental.jitclass(insideResultSpec)
class IsInsideResult(object):
    def __init__(self):
        self.isInside = False
