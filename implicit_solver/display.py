"""
@author: Vincent Bonnet
@description : Routine to display objects and constraints
"""

import matplotlib.pyplot as plt

def draw(data, frameId):
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }
    
    fig = plt.figure()
    plt.xlabel('x (in meters)')
    plt.ylabel('y (in meters)')
    plt.title('Mass-spring-damper - frame ' + str(frameId), fontdict=font)
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    
    # Draw constraints
    for constraint in data.constraints:
        ids = constraint.ids
        if (len(ids) >= 2):
            linedata = []
            for pid in ids:
                linedata.append(data.x[pid])
            x, y = zip(*linedata)
            ax.plot(x, y, 'k-', lw=1)
    
    # Draw particles
    x, y = zip(*data.x)
    ax.plot(x, y, 'go')
    
    plt.show()