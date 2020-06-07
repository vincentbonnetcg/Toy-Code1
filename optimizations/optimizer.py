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

        step = -gradient
        if NORMALIZED_STEP:
            step /= np.linalg.norm(step)
        step *= SCALE_STEP

        guess += step

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
def QuasiNewtonRaphson_BFGS(function):
    results = []
    guess = function.guess()
    results.append(np.copy(guess))

    terminate = False
    num_iterations = 0
    I = np.identity(2)
    H = np.identity(2) # approximate inverse hessian
    y = np.zeros(2) # gradient@x+1 - gradient@x

    while not terminate:
        gradient = function.gradient(guess)

        step = -H.dot(gradient)
        step *= SCALE_STEP

        guess += step

        # test termination conditions
        num_iterations += 1
        if np.linalg.norm(gradient) < THRESHOLD or num_iterations > MAX_ITERATIONS:
            terminate = True

        # update the inverse hessian matrix
        next_gradient = function.gradient(guess)
        y = next_gradient - gradient
        ys = np.inner(y, step) # scalar

        # early version
        #next_H = (I - np.outer(step, y) / ys)
        #next_H *= H
        #next_H *= (I - np.outer(y, step) / ys)
        #next_H += (np.outer(step, step) / ys)

        #optimized version
        Hy = np.dot(H, y) # vector
        yHy = np.inner(y, Hy) # scalar
        next_H = H
        next_H += ((ys+yHy) * np.outer(step, step) / ys ** 2)
        next_H -= (np.outer(Hy, step) + np.outer(step, Hy)) / ys

        H = next_H

        # store result
        results.append(np.copy(guess))

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

        step = -function.inv_hessian(guess).dot(gradient)
        step *= SCALE_STEP

        guess += step

        # test termination conditions
        num_iterations += 1
        if np.linalg.norm(gradient) < THRESHOLD or num_iterations > MAX_ITERATIONS:
            terminate = True

        # store result
        results.append(np.copy(guess))

    return results

