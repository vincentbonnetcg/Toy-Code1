"""
@author: Vincent Bonnet
@description :  Numerical differentiation
"""

import numpy as np

STENCIL_SIZE = 1e-6

# Private function to compute the gradient of a function with n arguments
# Used by numericalJacobian function
def _numericalGradient(function, argumentId, stencils, *args):
    argsListR = list(args)
    argsListL = list(args)   
    argsListR[argumentId] = np.add(args[argumentId], stencils)
    argsListL[argumentId] = np.subtract(args[argumentId], stencils)
    valueR = function(*argsListR)
    valueL = function(*argsListL)
    gradient = (valueR - valueL) / (STENCIL_SIZE * 2.0)
    return gradient
    
# This function returns a jacobian matrix with the following dimension :
# [function codomain dimension, function domain dimension]
# 'Function codomain dimension' : dimension of the function output
# 'Function domain dimension' : dimension of the input argumentId of the function
# Warning : Only use scalar as argument (no integer/boolean)
def numericalJacobian(function, argumentId, *args):
    functionCodomainDimension = 1
    functionDomainDimension = 1

    gradientList = []

    # compute gradients    
    if (np.isscalar(args[argumentId])):
        functionDomainDimension = 1        
        gradient = _numericalGradient(function, argumentId, STENCIL_SIZE, *args)        
        gradientList.append(gradient)
    else:
        functionDomainDimension = len(args[argumentId])
        stencils = np.zeros(functionDomainDimension)
        for i in range(functionDomainDimension):
            stencils.fill(0)
            stencils[i] = STENCIL_SIZE
            gradient = _numericalGradient(function, argumentId, stencils, *args)           
            gradientList.append(gradient)

    # assemble jacobian from gradients
    if len(gradientList)>0:
        if (np.isscalar(gradientList[0])):
            functionCodomainDimension = 1
        else:
            functionCodomainDimension = len(gradientList[0])

        jacobian = np.zeros(shape=(functionCodomainDimension, functionDomainDimension))
        for gradientId in range(len(gradientList)):
            jacobian[0:functionCodomainDimension, gradientId] = gradientList[gradientId]
    
        return jacobian            
