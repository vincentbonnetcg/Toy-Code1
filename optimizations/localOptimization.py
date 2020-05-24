"""
@author: Vincent Bonnet
@description : multivariable optimizations - gradient descent
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''
 Global Constants
'''
# Step parameter
NORMALIZED_STEP = True  # Only Gradient Descent
SCALE_STEP = 0.02 # Newton Method and Gradient Descent
# Termination condition
MAX_ITERATIONS = 200
THRESHOLD = 1e-07

'''
 1D Function For Example
'''
class function1D():
    def guess():
        return np.array(3.0)

    def value(X):
        return np.sin(X)

    def gradient(X):
        return np.array(np.cos(X))

    def inv_hessian(X):
        return np.array(1.0 / np.sin(X) * -1.0)
    
    def ranges():
        return (2.1, -2.1), (2.1, 2.1)

'''
 2D Function For Example
'''
class function2D():
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
        return (2.1, -2.1), (2.1, 2.1)

'''
 Gradient Descent
 Minimize the function
'''
def gradientDescent(function):
    results = []
    guess = function.guess()

    terminate = False

    previous_value = function.value(guess)
    num_iterations = 0

    # store first guess
    result = np.append(guess, previous_value)
    results.append(result)

    while not terminate:
        step = function.gradient(guess)
        if NORMALIZED_STEP:
            step /= np.linalg.norm(step)

        step *= SCALE_STEP
        guess -= step

        # test termination conditions
        value = function.value(guess)
        diff = np.linalg.norm(value - previous_value)
        previous_value = value
        num_iterations += 1

        if diff < THRESHOLD or num_iterations > MAX_ITERATIONS:
            terminate = True

        # store result
        result = np.append(guess, previous_value)
        results.append(result)

    return results

'''
 Newton Iteration with optimization
 Newton's method for unconstrained optimization finds local extrema (minima or maxima)
'''
def NewtonRaphson(function):
    results = []
    guess = function.guess()

    terminate = False

    num_iterations = 0

    # store first guess
    result = np.append(guess, function.value(guess))
    results.append(result)

    while not terminate:
        step = function.inv_hessian(guess).dot(function.gradient(guess)) * SCALE_STEP
        guess -= step

        # test termination conditions
        num_iterations += 1

        if np.linalg.norm(step) < THRESHOLD or num_iterations > MAX_ITERATIONS:
            terminate = True

        # store result
        result = np.append(guess, function.value(guess))
        results.append(result)

    return results


'''
 Show Result
'''
FONT = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

'''
 Show Result
'''
def draw1D(optimiser):

    def interface1D(X):
        return function1D.value(X)

    plt.title('1D function - ' + optimiser.__name__, fontdict=FONT)
    plt.xlabel('(x)')
    plt.ylabel('f(x)')

    # display function
    t = np.arange(0., 10., 0.2)
    plt.plot(t, interface1D(t), '-.', color="blue")

    # display result
    results = optimiser(function1D)
    X, Y = zip(*results)
    plt.plot(X, Y, '*', color="red")
    plt.show()

'''
 Show Result 2D
'''
def draw2D(optimiser):

    def interface2D(X, Y):
        return function2D.value([X, Y])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(65, 60)

    # display function
    min_x, max_x = function2D.ranges()
    X = np.arange(-2.1, 2.1, 0.1)
    Y = np.arange(-2.1, 2.1, 0.1)
    X, Y = np.meshgrid(X, Y)
    ax.plot_surface(X, Y, interface2D(X, Y), rstride=1, cstride=1, cmap='winter',alpha=0.80, antialiased=True)

    # display result
    results = optimiser(function2D)
    X, Y, Z = zip(*results)
    ax.scatter3D(X, Y, Z, c=[1,0,0], alpha=1.0, s=2)

    plt.title('2D function - ' + optimiser.__name__, fontdict=FONT)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    plt.show()

def main():
    #draw1D(gradientDescent)
    #draw2D(gradientDescent)
    #draw1D(NewtonRaphson)
    draw2D(NewtonRaphson)


if __name__ == '__main__':
    main()