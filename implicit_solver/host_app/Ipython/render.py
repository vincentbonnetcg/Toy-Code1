"""
@author: Vincent Bonnet
@description : Routine to display objects and constraints
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from tools import profiler
import numpy as np
import core.node_accessor as na

class Render:

    def __init__(self):
        '''
        Initialize Render
        '''
        self.fig = plt.figure()
        self.font = {'color':  'darkblue',
                     'weight': 'normal',
                     'size': 18}
        self.render_folder_path = ""
        self.ax = None
        self.min = [-5.0, -5.0]
        self.max = [5.0, 5.0]

    def set_viewport_limit(self, min_x, min_y, max_x, max_y):
        '''
        Specify the viewport limit
        '''
        self.min[0] = min_x
        self.min[1] = min_y
        self.max[0] = max_x
        self.max[1] = max_y

    def set_render_folder_path(self, path):
        '''
        Set the folder location to store image files
        '''
        self.render_folder_path = path

    def render_scene(self, scene, frame_id):
        '''
        Render the scene into a figue
        '''
        # Reset figure and create subplot
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('equal')
        self.ax.set_xlim(self.min[0], self.max[0])
        self.ax.set_ylim(self.min[1], self.max[1])

        # Statistics for legend
        stats_total_constraints = 0
        stats_total_nodes = 0

        # Set label
        plt.title('Implicit Solver - frame ' + str(frame_id), fontdict = self.font)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        # Draw constraints
        for condition in scene.conditions:
            num_constraints = condition.num_constraints()
            stats_total_constraints += num_constraints
            render_prefs = condition.meta_data.get("render_prefs" , None)
            if render_prefs is None:
                continue

            segs = []
            node_ids = condition.data.node_ids
            for ct_index in range(num_constraints):
                num_nodes = len(node_ids[ct_index])
                if num_nodes == 2:
                    points = []
                    for node_index in range (num_nodes):
                        x, v = na.node_state(scene, node_ids[ct_index][node_index])
                        points.append(x)
                    segs.append(points)

            line_segments = LineCollection(segs,
                                           linewidths=render_prefs['width'],
                                           colors=render_prefs['color'],
                                           linestyles=render_prefs['style'],
                                           alpha=render_prefs['alpha'])

            self.ax.add_collection(line_segments)

        # Draw nodes
        for dynamic in scene.dynamics:
            stats_total_nodes += len(dynamic.data)
            render_prefs = dynamic.meta_data.get("render_prefs" , None)
            if render_prefs is None:
                continue

            x, y = zip(*dynamic.x)
            self.ax.plot(x, y, '.', alpha=render_prefs['alpha'], color=render_prefs['color'], markersize = render_prefs['width'])

        # Draw kinematics
        for kinematic in scene.kinematics:
            render_prefs = kinematic.meta_data.get("render_prefs" , None)
            if render_prefs is None:
                continue

            vertices = kinematic.get_vertices(False)
            polygon  = patches.Polygon(vertices, facecolor=render_prefs['color'], alpha=render_prefs['alpha'])
            self.ax.add_patch(polygon)

        # Add Legend
        red_patch = patches.Patch(color='red', label=str(stats_total_nodes) + ' nodes')
        blue_patch = patches.Patch(color='blue', label=str(stats_total_constraints) + ' constraints')
        plt.legend(handles=[red_patch, blue_patch], loc='lower left')
        plt.show()

    def render_sparse_matrix(self, solver, frameId):
        '''
        Render the sparse matrix
        '''
        if (solver.A is not None):
            dense_A = np.abs(solver.A.todense())
            plt.imshow(dense_A, interpolation='none', cmap='binary')
            plt.colorbar()
        plt.show()

    @profiler.timeit
    def show_current_frame(self, dispatcher, frame_id):
        '''
        Display the current frame
        '''
        #self.fig = plt.figure(figsize=(7, 4), dpi=200) # to export higher resolution images
        scene = dispatcher.run("get_scene")
        self.fig = plt.figure()
        self.render_scene(scene, frame_id)
        #self.render_sparse_matrix(solver, frameId)

    @profiler.timeit
    def export_current_frame(self, filename):
        '''
        Export current frame into an image file
        '''
        if len(filename) > 0 and len(self.render_folder_path) > 0:
            self.fig.savefig(self.render_folder_path + "/" + filename)
