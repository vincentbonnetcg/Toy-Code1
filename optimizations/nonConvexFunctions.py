"""
@author: Vincent Bonnet
@description : test functions for optimization with non-convex functions
"""

import numpy as np

'''
 2D Function For Example
'''
class trigonometry2D:
    def guess():
        return np.array([0.5,0.6])

    def value(X):
        exp = np.exp(-X[0]**2-X[1]**2)
        return 0.75 * X[0] * exp

    def gradient(X):
        exp = np.exp(-X[0]**2-X[1]**2)
        dfdx = 0.75 * (exp - 2*exp*X[0]**2)
        dfdy = -1.5 * exp * X[0] * X[1]
        return np.array([dfdx, dfdy])

    def inv_hessian(X):
        exp = np.exp(-X[0]**2-X[1]**2)
        dfdxx = 0.75 * (4*exp*X[0]**3 - 6*exp*X[0])
        dfdxy = 0.75 * (-2*exp*X[1] + 4*exp*X[0]**2*X[1])
        dfdyy = -1.5 * X[0] * (-2*exp*X[1]**2 + exp)
        hessian = np.zeros((2,2))
        hessian[0][0] = dfdxx
        hessian[1][1] = dfdyy
        hessian[0][1] = dfdxy
        hessian[1][0] = dfdxy
        return np.linalg.inv(hessian)

    def ranges():
        return (-2.1, -2.1), (2.1, 2.1)
