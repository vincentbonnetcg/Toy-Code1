"""
@author: Vincent Bonnet
@description : Routine to display objects and constraints
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections

import lib.common as cm

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

    def render_objects(self, dispatcher, frame_id):
        '''
        Render the scene into a figure
        '''
        dynamics = dispatcher.get_dynamics()
        conditions = dispatcher.get_conditions()
        kinematics = dispatcher.get_kinematics()
        # Reset figure and create subplot
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        #self.ax.axis('equal') # do not use : it resizes the viewport during simulation
        self.ax.margins(0.05)
        #self.ax.set_aspect('equal') # break
        self.ax.autoscale(enable=False)
        fig_size = self.fig.get_size_inches()
        ratio = fig_size[0] / fig_size[1]
        width = self.max[0]-self.min[0]
        height = self.max[1]-self.min[1]
        expected_width = height * ratio
        offset = (expected_width - width) / 2
        self.ax.set_xlim(self.min[0]-offset, self.max[0]+offset)
        self.ax.set_ylim(self.min[1], self.max[1])
        # Statistics for legend
        total_constraints = 0
        total_nodes = 0
        total_node_blocks = 0
        total_constraint_blocks = 0

        # Set label
        plt.title(f'Implicit Solver - frame {frame_id}', fontdict = self.font)
        #plt.xlabel('x (m)')
        #plt.ylabel('y (m)')

        # Draw constraints
        for name in conditions:
            metadata = dispatcher.get_metadata(obj=name)
            total_constraints += metadata['num_constraints']
            total_constraint_blocks += metadata['num_blocks']
            render_prefs = metadata.get("render_prefs" , None)
            if render_prefs is None:
                continue

            segs = dispatcher.get_segments_from_constraint(condition=name)
            line_segments = collections.LineCollection(segs,
                                           linewidths=render_prefs['width'],
                                           colors=render_prefs['color'],
                                           linestyles=render_prefs['style'],
                                           alpha=render_prefs['alpha'])

            self.ax.add_collection(line_segments)

        # Draw nodes
        for name in dynamics:
            metadata = dispatcher.get_metadata(obj=name)
            total_nodes += metadata['num_nodes']
            total_node_blocks += metadata['num_blocks']
            render_prefs = metadata.get("render_prefs" , None)
            if render_prefs is None:
                continue

            dynamic_data = dispatcher.get_nodes_from_dynamic(dynamic=name)
            x, y = zip(*dynamic_data)
            self.ax.plot(x, y, '.', alpha=render_prefs['alpha'],
                                     color=render_prefs['color'],
                                     markersize = render_prefs['width'])

        # Draw kinematics
        for name in kinematics:
            metadata = dispatcher.get_metadata(obj=name)
            render_prefs = metadata.get("render_prefs" , None)
            if render_prefs is None:
                continue

            normals = dispatcher.get_normals_from_kinematic(kinematic=name)
            line_normals = collections.LineCollection(normals,
                                           linewidths=1,
                                           colors=render_prefs['color'],
                                           alpha=render_prefs['alpha'])

            self.ax.add_collection(line_normals)

            triangles = []
            shape = dispatcher.get_shape_from_kinematic(kinematic=name)
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
        node_patch = patches.Patch(color='blue', label=f'{total_nodes} nodes')
        nblock_patch = patches.Patch(color='lightblue', label= f'{total_node_blocks} node blocks')
        cnt_patch = patches.Patch(color='green', label=f'{total_constraints} constraints')
        cblock_patch = patches.Patch(color='lightgreen', label=f'{total_constraint_blocks} constraint blocks')
        plt.legend(handles=[node_patch, nblock_patch, cnt_patch, cblock_patch], loc='lower left')
        plt.show()

    def render_sparse_matrix(self, dispatcher, frameId):
        '''
        Render the sparse matrix
        '''
        dense_A = dispatcher.get_sparse_matrix_as_dense(as_binary=True)
        if dense_A is None:
            return

        plt.imshow(dense_A, interpolation='none', cmap='binary')
        plt.show()

    @cm.timeit
    def show_current_frame(self, dispatcher, frame_id):
        '''
        Display the current frame
        '''
        #self.fig = plt.figure(figsize=(7, 4), dpi=200) # to export higher resolution images
        self.fig = plt.figure()
        self.render_objects(dispatcher, frame_id)
        #self.render_sparse_matrix(dispatcher, frame_id)

    @cm.timeit
    def export_current_frame(self, filename):
        '''
        Export current frame into an image file
        '''
        if len(filename) > 0 and len(self.render_folder_path) > 0:
            self.fig.savefig(self.render_folder_path + "/" + filename)
