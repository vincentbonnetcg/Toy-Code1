"""
@author: Vincent Bonnet
@description : test functions for optimization with convex functions
"""

import numpy as np

'''
 BohachevskyN1 function
'''
class BohachevskyN1:
    def guess():
        return np.array([-50.,-50.])

    def value(X):
        result = X[0]**2 + 2*X[1]**2
        result -= 0.3*np.cos(3*np.pi*X[0])
        result -= 0.4*np.cos(4*np.pi*X[1])
        result += 0.7
        return result

    def gradient(X):
        dfdx = 2.*X[0] + (0.3*3.*np.pi*np.sin(3*np.pi*X[0]))
        dfdy = 4.*X[1] + (0.4*4.*np.pi*np.sin(4*np.pi*X[1]))
        return np.array([dfdx, dfdy])

    def inv_hessian(X):
        dfdxx = 2.+((3.*np.pi)**2)*0.3*np.cos(3*np.pi*X[0])
        dfdyy = 4.+((4.*np.pi)**2)*0.4*np.cos(4*np.pi*X[1])
        dfdxy = 0
        hessian = np.zeros((2,2))
        hessian[0][0] = dfdxx
        hessian[1][1] = dfdyy
        hessian[0][1] = dfdxy
        hessian[1][0] = dfdxy
        return np.linalg.inv(hessian)

    def ranges():
        return (-100, -100), (100, 100)

class McCormick:
    def guess():
        return np.array([4.0,-2.0])

    def value(X):
        result = np.sin(X[0]+X[1])
        result += (X[0]-X[1])**2.0
        result += -1.5*X[0] + 2.5*X[1] + 1.0
        return result

    def gradient(X):
        dfdx = np.cos(X[0]+X[1]) + 2*(X[0]-X[1]) - 1.5
        dfdy = np.cos(X[0]+X[1]) - 2*(X[0]-X[1]) + 2.5
        return np.array([dfdx, dfdy])

    def inv_hessian(X):
        dfdxx = -1.*np.sin(X[0]+X[1]) + 2
        dfdyy = -1.*np.sin(X[0]+X[1]) + 2
        dfdxy = -1.*np.sin(X[0]+X[1]) - 2
        hessian = np.zeros((2,2))
        hessian[0][0] = dfdxx
        hessian[1][1] = dfdyy
        hessian[0][1] = dfdxy
        hessian[1][0] = dfdxy
        return np.linalg.inv(hessian)

    def ranges():
        return (-1.5, -3.), (4., 3.)

class Booth:
    def guess():
        return np.array([-10.0,-10.0])

    def value(X):
        a = X[0] + 2*X[1] - 7
        b = 2*X[0] + X[1] - 5
        result = a**2 + b**2
        return result

    def gradient(X):
        dfdx = 10*X[0] + 8*X[1] - 34
        dfdy = 8*X[0] + 10*X[1] - 38
        return np.array([dfdx, dfdy])

    def inv_hessian(X):
        dfdxx = 10.
        dfdyy = 10.
        dfdxy = 8.
        hessian = np.zeros((2,2))
        hessian[0][0] = dfdxx
        hessian[1][1] = dfdyy
        hessian[0][1] = dfdxy
        hessian[1][0] = dfdxy
        return np.linalg.inv(hessian)

    def ranges():
        return (-10, -10), (10, 10)

