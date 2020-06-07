"""
@author: Vincent Bonnet
@description : render functions
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

FONT = {'family': 'serif',
        'color':  'darkblue',
        'weight': 'normal',
        'size': 14,
        }

def draw2D(function, optimiser):

    interface2D = lambda x, y : function.value([x, y])

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.view_init(65, 60)

    # display function
    subdivisions = 40
    min_arg, max_arg = function.ranges()
    X = np.linspace(min_arg[0], max_arg[0], subdivisions)
    Y = np.linspace(min_arg[1], max_arg[1], subdivisions)
    X, Y = np.meshgrid(X, Y)
    Z = interface2D(X, Y)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='winter', alpha=0.5)
    ax.contour(X, Y, Z, 20, colors='black')

    title = function.__name__
    # display optimizer result
    if optimiser:
        results = np.asarray(optimiser(function))
        X, Y = results[:,0],results[:,1]
        Z = function.value(np.asarray([X,Y]))
        color = np.linspace([1,0,0], [0,1,0], num=len(results))
        ax.scatter3D(X, Y, Z, c=color, alpha=1.0, s=3)
        title += ' - ' + optimiser.__name__
        title += ' - iter(' + str(len(results)-1) + ')'

    plt.title(title, fontdict=FONT)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)')
    plt.show()