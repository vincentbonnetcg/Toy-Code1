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
MAX_ITERATIONS = 50
NORMALIZED_STEP = True  # Used when line search is False
SCALE_STEP = 0.1

'''
 1D Function For Example
'''
class function1D():
    def guess():
        return 2.0

    def value(X):
        return np.sin(X)

    def gradient(X):
        return np.cos(X)

    def hessian(X):
        return -1.0 * np.sin(X)

'''
 2D Function For Example
'''
class function2D():
    def guess():
        return np.array([0.5,0.6])

    def value(X):
        return X[0]*np.exp(-X[0]**2-X[1]**2)*0.75

    def gradient(X):
        dfdx = (6*X[0]**2 - 3)*np.exp(-X[0]**2-X[1]**2) * -0.25
        dfdy = (3*X[0]*X[1])*np.exp(-X[1]**2-X[0]**2) * -3/2
        return np.array([dfdx, dfdy])

    def hessian(X):
        # TODO
        return None

'''
 Gradient Descent
'''
def gradientDescent(function):
    results = []
    guess = function.guess()

    for i in range(MAX_ITERATIONS):
        # store result
        result = np.append(guess, function.value(guess))
        results.append(result)

        # gradient descent
        step = function.gradient(guess)
        if NORMALIZED_STEP:
            step /= np.linalg.norm(step)

        step *= -SCALE_STEP
        guess += step

    return results

'''
 Newton Iteration with optimization
'''
def NewtonRaphson(function):
    results = []
    guess = function.guess()

    for i in range(MAX_ITERATIONS):
        # store result
        result = np.append(guess, function.value(guess))
        results.append(result)

        # store result
        result = np.append(guess, function.value(guess))
        results.append(result)

        # Newton Raphson
        guess -= function.value(guess) / function.gradient(guess)

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
    draw1D(gradientDescent)
    draw2D(gradientDescent)
    #draw1D(NewtonRaphson)
    #draw2D(NewtonRaphson)


if __name__ == '__main__':
    main()