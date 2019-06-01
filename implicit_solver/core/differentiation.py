"""
@author: Vincent Bonnet
@description :  Generic Numerical differentiation
Slow and only just for testing
"""

# Reference #
# Fornberg, Bengt. "Generation of finite difference formulas on arbitrarily spaced grids."
# Mathematics of computation 51, no. 184 (1988): 699-706.

# Examples #
#force = diff.numerical_jacobian(elastic_spring_energy, 0, x0, x1, rest_length, stiffness) * -1.0
#dfdx = diff.numerical_jacobian(spring_stretch_force, 0, x0, x1, rest_length, stiffness)
#dfdv = diff.numerical_jacobian(spring_damping_force, 2, x0, x1, v0, v1, damping)

import numpy as np

# Central Difference tables and accuracy order
STENCIL_SIZE = 1e-6
FIRST_DERIVATIVE_OFFSET = [[-STENCIL_SIZE, STENCIL_SIZE],
                           [-2.0 * STENCIL_SIZE, -STENCIL_SIZE, STENCIL_SIZE, 2.0 * STENCIL_SIZE],
                           [-3.0 * STENCIL_SIZE, -2.0 * STENCIL_SIZE, -STENCIL_SIZE, STENCIL_SIZE, 2.0 * STENCIL_SIZE, 3.0 * STENCIL_SIZE]]

FIRST_DERIVATIVE_COEFS = [[-0.5, 0.5],
                          [1/12, -2/3, 2/3, -1/12],
                          [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60]]

# 0 means accuracy order of 2
# 1 means accuracy order of 4
# 2 means accuracy order of 6
ACCURACY_ORDER_INDEX = 0

def numerical_jacobian(function, argument_id, *args):
    '''
    Returns a jacobian matrix of dimension :
    [function codomain dimension, function domain dimension]
    'Function codomain dimension' : dimension of the function output
    'Function domain dimension' : dimension of the input argument_id of the function
    Warning : Only use scalar as argument (no integer/boolean)
    Note : This function can be used to compute the gradient of function
    '''
    function_codomain_dimension = 1 # default when np.isscalar(args[argument_id])
    function_domain_dimension = 1 # default when np.isscalar(gradient_list[0])
    gradient_list = []

    # Differentiation settings
    offsets = FIRST_DERIVATIVE_OFFSET[ACCURACY_ORDER_INDEX]
    coefs = FIRST_DERIVATIVE_COEFS[ACCURACY_ORDER_INDEX]

    # Copy the arguments of the function
    # This is a shallow copy by consequence the numerical differientiation
    # will have to do a deep copy for every modified argument
    args_array = np.array(args)

    if np.isscalar(args[argument_id]):
        function_domain_dimension = 1
        # deepcopy of the argument to be modified
        args_array[argument_id] = np.copy(args[argument_id])

        stencils = offsets + args_array[argument_id]

        args_array[argument_id] = stencils[0]
        gradient = function(*args_array) * coefs[0]
        for i in range(1, len(stencils)):
            args_array[argument_id] = stencils[i]
            gradient += function(*args_array) * coefs[i]

        gradient /= STENCIL_SIZE
        gradient_list.append(gradient)
    else:
        function_domain_dimension = len(args[argument_id])
        for component_id in range(function_domain_dimension):
            # deepcopy of the argument to be modified
            args_array[argument_id] = np.copy(args[argument_id])

            stencils = offsets + args_array[argument_id][component_id]

            args_array[argument_id][component_id] = stencils[0]
            gradient = function(*args_array) * coefs[0]
            for i in range(1, len(stencils)):
                args_array[argument_id][component_id] = stencils[i]
                gradient += function(*args_array) * coefs[i]

            gradient /= STENCIL_SIZE
            gradient_list.append(gradient)

    # assemble jacobian/gradient from gradients
    if np.isscalar(gradient_list[0]):
        gradient = np.asarray(gradient_list)
        return gradient

    function_codomain_dimension = len(gradient_list[0])
    jacobian = np.zeros(shape=(function_codomain_dimension, function_domain_dimension))
    for gradient_id, gradient in enumerate(gradient_list):
        jacobian[0:function_codomain_dimension, gradient_id] = gradient

    return jacobian

def numerical_hessian(function, first_argument_id, second_argument_id, *args):
    '''
    Returns the Hessian by using two consecutive derivations (mixed derivatives)
    The order of differientation doesn't matter : see Clairaut's theorem/Schwarz's theorem
    '''
    class DifferentiationWrapper:
        def __init__(self, function, argument_id):
            self.function = function
            self.argument_id = argument_id

        def numerical_jacobian(self, *args):
            return numerical_jacobian(self.function, self.argument_id, *args)

    jacobian_wrapper = DifferentiationWrapper(function, first_argument_id)

    return numerical_jacobian(jacobian_wrapper.numerical_jacobian, second_argument_id, *args)
