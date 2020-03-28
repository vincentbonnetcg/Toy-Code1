"""
@author: Vincent Bonnet
@description : Routine to display objects and constraints
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections

import lib.common as cm
from lib.system import Scene

import numpy as np

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

    def render_scene(self, dispatcher, scene : Scene, frame_id):
        '''
        Render the scene into a figue
        '''
        # Reset figure and create subplot
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.ax.axis('equal')
        self.ax.margins(0.05)
        #self.ax.set_aspect('equal')
        self.ax.autoscale(enable=False)
        self.ax.set_xlim(self.min[0], self.max[0])
        self.ax.set_ylim(self.min[1], self.max[1])

        # Statistics for legend
        stats_total_constraints = 0
        stats_total_nodes = 0
        stats_avg_block_per_objects = 0
        stats_avg_block_per_constraints = 0

        # Set label
        plt.title('Implicit Solver - frame ' + str(frame_id), fontdict = self.font)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')

        # Draw constraints
        for condition_id, condition in enumerate(scene.conditions):
            stats_total_constraints += condition.num_constraints()
            stats_avg_block_per_constraints += condition.num_blocks()
            render_prefs = condition.meta_data.get("render_prefs" , None)
            if render_prefs is None:
                continue

            segs = dispatcher.run('get_segments_from_constraint', index=condition_id)
            line_segments = collections.LineCollection(segs,
                                           linewidths=render_prefs['width'],
                                           colors=render_prefs['color'],
                                           linestyles=render_prefs['style'],
                                           alpha=render_prefs['alpha'])

            self.ax.add_collection(line_segments)

        if stats_avg_block_per_constraints > 0:
            stats_avg_block_per_constraints /= len(scene.conditions)
            stats_avg_block_per_constraints = round(stats_avg_block_per_constraints, 2)

        # Draw nodes
        for dynamic_id, dynamic in enumerate(scene.dynamics):
            stats_total_nodes += dynamic.num_nodes()
            stats_avg_block_per_objects += dynamic.num_blocks()

            render_prefs = dynamic.meta_data.get("render_prefs" , None)
            if render_prefs is None:
                continue

            dynamic_data = dispatcher.run('get_nodes_from_dynamic', index=dynamic_id)
            x, y = zip(*dynamic_data)
            self.ax.plot(x, y, '.', alpha=render_prefs['alpha'],
                                     color=render_prefs['color'],
                                     markersize = render_prefs['width'])

        stats_avg_block_per_objects /= len(scene.dynamics)
        stats_avg_block_per_objects = round(stats_avg_block_per_objects, 2)

        # Draw kinematics
        for kinematic_id, kinematic in enumerate(scene.kinematics):
            render_prefs = kinematic.meta_data.get("render_prefs" , None)
            if render_prefs is None:
                continue

            triangles = []
            shape = dispatcher.run('get_shape_from_kinematic', index=kinematic_id)
            for face_id in shape.face:
                v0 = shape.vertex[face_id[0]]
                v1 = shape.vertex[face_id[1]]
                v2 = shape.vertex[face_id[2]]
                triangles.append([v0, v1, v2])

            collec = collections.PolyCollection(triangles, facecolors=render_prefs['color'],
                                                            edgecolors=render_prefs['color'],
                                                            alpha=render_prefs['alpha'])
            self.ax.add_collection(collec)

        # Add Legend
        red_patch = patches.Patch(color='red', label=str(stats_total_nodes) + ' nodes')
        blue_patch = patches.Patch(color='blue', label=str(stats_total_constraints) + ' constraints')
        green_patch = patches.Patch(color='green', label=str(stats_avg_block_per_objects) + ' avg block/obj')
        lgreen_patch = patches.Patch(color='lightgreen', label=str(stats_avg_block_per_constraints) + ' avg block/cts')
        plt.legend(handles=[red_patch, blue_patch, green_patch, lgreen_patch], loc='lower left')
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

    @cm.timeit
    def show_current_frame(self, dispatcher, frame_id):
        '''
        Display the current frame
        '''
        #self.fig = plt.figure(figsize=(7, 4), dpi=200) # to export higher resolution images
        scene = dispatcher.run("get_scene")
        self.fig = plt.figure()
        self.render_scene(dispatcher, scene, frame_id)
        #self.render_sparse_matrix(solver, frameId)

    @cm.timeit
    def export_current_frame(self, filename):
        '''
        Export current frame into an image file
        '''
        if len(filename) > 0 and len(self.render_folder_path) > 0:
            self.fig.savefig(self.render_folder_path + "/" + filename)
