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

# function : function to derivate
# Accuracy Order can be 2, 4, 6
def numericalGradient(function, argumentId, componentId, *args):
    accuracyOrderIndex = (int)(ACCURACY_ORDER / 2) - 1
    offsets = FIRST_DERIVATIVE_OFFSET[accuracyOrderIndex]
    coefs = FIRST_DERIVATIVE_COEFS[accuracyOrderIndex]

    argsList = list(args)
    array = argsList
    valueId = argumentId
    if (not np.isscalar(args[argumentId])):
        array = argsList[argumentId]
        valueId = componentId

    gradient = None
    stencils = np.add(offsets, array[valueId])
    for i in range(len(stencils)):
        array[valueId] = stencils[i]
        if (gradient is None):
            gradient = np.multiply(function(*argsList), coefs[i])
        else:
            gradient += np.multiply(function(*argsList), coefs[i])

    gradient /= STENCIL_SIZE
    return gradient

# This function returns a jacobian matrix with the following dimension :
# [function codomain dimension, function domain dimension]
# 'Function codomain dimension' : dimension of the function output
# 'Function domain dimension' : dimension of the input argumentId of the function
# Warning : Only use scalar as argument (no integer/boolean)
def numericalJacobian(function, argumentId, *args):
    functionCodomainDimension = 1 # default when np.isscalar(args[argumentId])
    functionDomainDimension = 1 # default when np.isscalar(gradientList[0])

    gradientList = []

    # compute gradients for each component of the argument
    if (not np.isscalar(args[argumentId])):
        functionDomainDimension = len(args[argumentId])

    for componentId in range(functionDomainDimension):
        gradient = numericalGradient(function, argumentId, componentId, *args)
        gradientList.append(gradient)

    # assemble jacobian from gradients
    if len(gradientList)>0:
        if (not np.isscalar(gradientList[0])):
            functionCodomainDimension = len(gradientList[0])

        jacobian = np.zeros(shape=(functionCodomainDimension, functionDomainDimension))
        for gradientId in range(len(gradientList)):
            jacobian[0:functionCodomainDimension, gradientId] = gradientList[gradientId]

        return jacobian
