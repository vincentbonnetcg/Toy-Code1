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

ACCURACY_ORDER = 2


def numericalDifferentiation(function, argumentId, componentId, *args):
    '''
    function to derivate !!!
    Accuracy Order can be 2, 4, 6
    '''
    accuracyOrderIndex = (int)(ACCURACY_ORDER / 2) - 1
    offsets = FIRST_DERIVATIVE_OFFSET[accuracyOrderIndex]
    coefs = FIRST_DERIVATIVE_COEFS[accuracyOrderIndex]

    #argsList = copy.deepcopy(args) # expensive operation => replaced with lines below
    argsList = []
    for a in args:
        argsList.append(np.copy(a))

    array = argsList
    valueId = argumentId
    if not np.isscalar(args[argumentId]):
        array = argsList[argumentId]
        valueId = componentId

    stencils = offsets + array[valueId]

    # compute gradients
    array[valueId] = stencils[0]
    gradient = function(*argsList) * coefs[0]

    for i in range(1, len(stencils)):
        array[valueId] = stencils[i]
        gradient += function(*argsList) * coefs[i]

    gradient /= STENCIL_SIZE
    return gradient

def numericalJacobian(function, argumentId, *args):
    '''
    This function returns a jacobian matrix with the following dimension :
    [function codomain dimension, function domain dimension]
    'Function codomain dimension' : dimension of the function output
    'Function domain dimension' : dimension of the input argumentId of the function
    Warning : Only use scalar as argument (no integer/boolean)
    Note : This function can be used to compute the gradient of function\
    '''
    functionCodomainDimension = 1 # default when np.isscalar(args[argumentId])
    functionDomainDimension = 1 # default when np.isscalar(gradientList[0])
    gradientList = []

    # compute gradients for each component of the argument
    if not np.isscalar(args[argumentId]):
        functionDomainDimension = len(args[argumentId])

    for componentId in range(functionDomainDimension):
        gradient = numericalDifferentiation(function, argumentId, componentId, *args)
        gradientList.append(gradient)

    # each gradient is a single value -> return a gradient
    if np.isscalar(gradientList[0]):
        gradient = np.asarray(gradientList)
        return gradient

    # assemble jacobian from gradients
    functionCodomainDimension = len(gradientList[0])
    jacobian = np.zeros(shape=(functionCodomainDimension, functionDomainDimension))
    for gradientId, gradient in enumerate(gradientList):
        jacobian[0:functionCodomainDimension, gradientId] = gradient

    return jacobian

def numericalHessian(function, argumentId0, argumentId1, *args):
    '''
    Return the Hessian by using two consecutive derivations (mixed derivatives)
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
