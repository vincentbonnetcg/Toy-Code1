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
MAX_ITERATIONS = 100
NORMALIZED_STEP = True  # Used when line search is False
SCALE_STEP = 0.1

'''
 1D Function For Example
'''
def initGuess1D():
    return [2.0]

def function1D(X):
    return np.sin(X[0])

def gradient1D(X):
    return [np.cos(X[0])]

'''
 2D Function For Example
'''
def initGuess2D():
    return (0.5,0.6)

def function2D(X):
    return X[0]*np.exp(-X[0]**2-X[1]**2)*0.75

def gradient2D(X):
    dx = (6*X[0]**2 - 3)*np.exp(-X[0]**2-X[1]**2) * -0.25
    dy = (3*X[0]*X[1])*np.exp(-X[1]**2-X[0]**2) * -3/2
    return (dx, dy)

'''
 Gradient Descent
''' 
def gradientDescent(init, function, gradient):
    results = []
    guess = init()
    
    for i in range(MAX_ITERATIONS):
        # store result
        z = function(guess) + 0.05
       
        result = np.append(guess, z)
        results.append(result)
        
        # gradient descent
        step = gradient(guess)
        if NORMALIZED_STEP:
            step /= np.linalg.norm(step)

        step *= -SCALE_STEP
        guess = np.add(guess, step)
        
    return results

'''
 Show Result 1D
'''
def interface1D(X):
    return function1D([X])

# prepare display
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
plt.title('1D function - Gradient Descent', fontdict=font)
plt.xlabel('(x)')
plt.ylabel('f(x)')

# display function
t = np.arange(0., 10., 0.2)
plt.plot(t, interface1D(t), '-.', color="blue")

# display result
results = gradientDescent(initGuess1D, function1D, gradient1D)
X, Y = zip(*results)
plt.plot(X, Y, '*', color="red")
plt.show()

'''
 Show Result 2D
'''
def interface2D(X, Y):
    return function2D((X, Y))

# prepare display
fig = plt.figure()
ax = Axes3D(fig)
ax.view_init(65, 60)

# display function
X = np.arange(-2.1, 2.1, 0.1)
Y = np.arange(-2.1, 2.1, 0.1)
X, Y = np.meshgrid(X, Y)
ax.plot_surface(X, Y, interface2D(X, Y), rstride=1, cstride=1, cmap='winter',alpha=0.80, antialiased=True)

# display result
results = gradientDescent(initGuess2D, function2D, gradient2D)
X, Y, Z = zip(*results)
ax.scatter3D(X, Y, Z, c=[1,0,0], alpha=1.0, s=2)

plt.title('2D function - Gradient Descent', fontdict=font)
plt.show()
