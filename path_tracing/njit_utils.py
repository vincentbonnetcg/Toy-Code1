"""
@author: Vincent Bonnet
@description : jitted utilities
"""

import numpy as np
import numba

#@numba.njit
def sphere_intersect(ray_o, ray_d, sphere_c, sphere_r):
    o = ray_o - sphere_c
    a = np.dot(ray_d, ray_d)
    b = np.dot(ray_d, o) * 2.0
    c = np.dot(o, o) - sphere_r**2
    # solve ax**2 + bx + c = 0
    dis = b**2 - 4*a*c  # discriminant

    if dis < 0.0:
        # no solution
        return -1.0

    if np.isclose(dis, 0.0):
        # one solution
        return -b / 2 * a

    # two solution
    sq = np.sqrt(dis)
    s1 = (-b-sq) / 2*a  # first solution
    s2 = (-b+sq) / 2*a # second solution

    if s1 < 0.0 and s2 < 0.0:
        return False

    t = s2
    if s1 > 0.0 and s2 > 0.0:
        t = np.minimum(s1, s2)
    elif s1 > 0.0:
        t = s1

    return t
