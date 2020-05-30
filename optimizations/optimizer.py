"""
@author: Vincent Bonnet
@description : multivariable optimizers
"""

import numpy as np

# Step parameter
NORMALIZED_STEP = True  # Only Gradient Descent
SCALE_STEP = 0.02 # Newton Method and Gradient Descent
# Termination condition
MAX_ITERATIONS = 200
THRESHOLD = 1e-07

'''
 Gradient Descent
'''
def GradientDescent(function):
    results = []
    guess = function.guess()
    results.append(np.copy(guess))

    terminate = False
    num_iterations = 0

    while not terminate:
        gradient = function.gradient(guess)

        step = gradient
        if NORMALIZED_STEP:
            step /= np.linalg.norm(step)

        step *= SCALE_STEP
        guess -= step

        # test termination conditions
        num_iterations += 1
        if np.linalg.norm(gradient) < THRESHOLD or num_iterations > MAX_ITERATIONS:
            terminate = True

        # store result
        results.append(np.copy(guess))

    return results

'''
 QuasiNewton optimization
'''
def QuasiNewtonRaphson(function):
    results = []
    guess = function.guess()
    results.append(np.copy(guess))

    terminate = False
    num_iterations = 0

    while not terminate:
        gradient = function.gradient(guess)

        # TODO

        terminate = True

    return results

'''
 Newton Iteration optimization
 Newton's method for unconstrained optimization finds local extrema (minima or maxima)
'''
def NewtonRaphson(function):
    results = []
    guess = function.guess()
    results.append(np.copy(guess))

    terminate = False
    num_iterations = 0

    while not terminate:
        gradient = function.gradient(guess)

        step = function.inv_hessian(guess).dot(gradient) * SCALE_STEP
        guess -= step

        # test termination conditions
        num_iterations += 1
        if np.linalg.norm(gradient) < THRESHOLD or num_iterations > MAX_ITERATIONS:
            terminate = True

        # store result
        results.append(np.copy(guess))

    return results

