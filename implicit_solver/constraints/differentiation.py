"""
@author: Vincent Bonnet
@description :  Numerical differentiation
"""

# Reference #
# Fornberg, Bengt. "Generation of finite difference formulas on arbitrarily spaced grids."
# Mathematics of computation 51, no. 184 (1988): 699-706.

import numpy as np

# Central Difference tables and accuracy order
STENCIL_SIZE = 1e-6
FIRST_DERIVATIVE_OFFSET = [[-STENCIL_SIZE, STENCIL_SIZE],
                           [-2.0 * STENCIL_SIZE, -STENCIL_SIZE, STENCIL_SIZE, 2.0 * STENCIL_SIZE],
                           [-3.0 * STENCIL_SIZE, -2.0 * STENCIL_SIZE, -STENCIL_SIZE, STENCIL_SIZE, 2.0 * STENCIL_SIZE, 3.0 * STENCIL_SIZE]]

FIRST_DERIVATIVE_COEFS = [[-0.5, 0.5],
                          [1/12, -2/3, 2/3, -1/12],
                          [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60]]

ACCURACY_ORDER = 2 # Accuracy Order can be 2, 4, 6

def numericalJacobian(function, argumentId, *args):
    '''
    Returns a jacobian matrix of dimension :
    [function codomain dimension, function domain dimension]
    'Function codomain dimension' : dimension of the function output
    'Function domain dimension' : dimension of the input argumentId of the function
    Warning : Only use scalar as argument (no integer/boolean)
    Note : This function can be used to compute the gradient of function
    '''
    functionCodomainDimension = 1 # default when np.isscalar(args[argumentId])
    functionDomainDimension = 1 # default when np.isscalar(gradientList[0])
    gradientList = []

    # Differentiation settings
    accuracyOrderIndex = (int)(ACCURACY_ORDER / 2) - 1
    offsets = FIRST_DERIVATIVE_OFFSET[accuracyOrderIndex]
    coefs = FIRST_DERIVATIVE_COEFS[accuracyOrderIndex]

    # Copy the arguments of the function
    # This is a shallow copy by consequence the numerical differientiation
    # will have to do a deep copy for every modified argument
    argsList = np.array(args)

    if np.isscalar(args[argumentId]):
        functionDomainDimension = 1
        # deepcopy of the argument to be modified
        argsList[argumentId] = np.copy(args[argumentId])

        stencils = offsets + argsList[argumentId]

        argsList[argumentId] = stencils[0]
        gradient = function(*argsList) * coefs[0]
        for i in range(1, len(stencils)):
            argsList[argumentId] = stencils[i]
            gradient += function(*argsList) * coefs[i]

        gradient /= STENCIL_SIZE
        gradientList.append(gradient)
    else:
        functionDomainDimension = len(args[argumentId])
        for componentId in range(functionDomainDimension):
            # deepcopy of the argument to be modified
            argsList[argumentId] = np.copy(args[argumentId])

            stencils = offsets + argsList[argumentId][componentId]

            argsList[argumentId][componentId] = stencils[0]
            gradient = function(*argsList) * coefs[0]
            for i in range(1, len(stencils)):
                argsList[argumentId][componentId] = stencils[i]
                gradient += function(*argsList) * coefs[i]

            gradient /= STENCIL_SIZE
            gradientList.append(gradient)

    # assemble jacobian/gradient from gradients
    if np.isscalar(gradientList[0]):
        gradient = np.asarray(gradientList)
        return gradient

    functionCodomainDimension = len(gradientList[0])
    jacobian = np.zeros(shape=(functionCodomainDimension, functionDomainDimension))
    for gradientId, gradient in enumerate(gradientList):
        jacobian[0:functionCodomainDimension, gradientId] = gradient

    return jacobian

def numericalHessian(function, argumentId0, argumentId1, *args):
    '''
    Returns the Hessian by using two consecutive derivations (mixed derivatives)
    The order of differientation doesn't matter : see Clairaut's theorem/Schwarz's theorem
    '''
    class diffHelper:
        def __init__(self, function, argumentId):
            self.function = function
            self.argumentId = argumentId

        def firstDerivative(self, *args):
            return numericalJacobian(self.function, self.argumentId, *args)

    derivative = diffHelper(function, argumentId0)

    return numericalJacobian(derivative.firstDerivative, argumentId1, *args)
