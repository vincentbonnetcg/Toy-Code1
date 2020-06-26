"""
@author: Vincent Bonnet
@description : render helper
"""

import matplotlib.pyplot as plt

class RenderHelper:

    def __init__(self, min_x, max_x, min_y, max_y):
        self.fig = plt.figure()
        self.font = {'family': 'serif',
                     'color':  'darkred',
                     'weight': 'normal',
                     'size': 12}
        self.output_folder = ""
        self.ax = None
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y

    def set_output_folder(self, folder):
        self.output_folder = folder

    def draw(self, data):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'draw'")

    def prepare_figure(self):
        self.fig = plt.figure()
        self.fig.clear()
        self.ax = self.fig.add_subplot(1, 1, 1)
        #self.ax.axis('equal') # FIXME - causes problem
        self.ax.autoscale(enable=False)
        fig_size = self.fig.get_size_inches()
        ratio = fig_size[0] / fig_size[1]
        width = self.max_x-self.min_x
        height = self.max_y-self.min_y
        expected_width = height * ratio
        offset = (expected_width - width) / 2
        self.ax.set_xlim(self.min_x-offset, self.max_x+offset)
        self.ax.set_ylim(self.min_y, self.max_y)

    def show_figure(self, data):
        self.draw(data)
        plt.show()

    def export_figure(self, filename):
        if len(filename) > 0 and len(self.output_folder) > 0:
            self.fig.savefig(self.output_folder + "/" + filename)
