"""
@author: Vincent Bonnet
@description : Routine to display objects and constraints
"""

import matplotlib.pyplot as plt
import profiler as profiler

class Render:
    
    def __init__(self):
        self.fig = plt.figure()
        self.font = {'family': 'serif',
                     'color':  'darkred',
                     'weight': 'normal',
                     'size': 12 }
        self.renderFolderPath = ""
    
    # Set where to save the files
    def setRenderFolderPath(self, path):
        self.renderFolderPath = path
    
    # Render in the current figure
    def _render(self, data, frameId):
        # Reset figure and create subplot
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('equal')
        self.ax.set_xlim(-5,5)
        self.ax.set_ylim(-5,5)
        
        # Set label
        plt.title('Mass-spring-damper - frame ' + str(frameId), fontdict=self.font)
        plt.xlabel('x (in meters)')
        plt.ylabel('y (in meters)')
        
        # Draw constraints
        for constraint in data.constraints:
            ids = constraint.ids
            if (len(ids) >= 2):
                linedata = []
                for pid in ids:
                    linedata.append(data.x[pid])
                x, y = zip(*linedata)
                self.ax.plot(x, y, 'k-', lw=1)
        
        # Draw particles
        x, y = zip(*data.x)
        self.ax.plot(x, y, 'go')

    # Draw and display single frame
    @profiler.timeit
    def showCurrentFrame(self, data, frameId):
        self.fig = plt.figure()
        self._render(data, frameId)
        plt.show()
        
    # Export frame
    @profiler.timeit
    def exportCurrentFrame(self, filename):
        if len(filename) > 0 and len(self.renderFolderPath) > 0:
            self.fig.savefig(self.renderFolderPath + "/" + filename)



