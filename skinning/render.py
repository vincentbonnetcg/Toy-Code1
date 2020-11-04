"""
@author: Vincent Bonnet
@description : Render Skinning
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

def draw(mesh, skeleton, weights, displacement, frame_id, render_folder_path = ""):
    '''
    Drawing function to display the mesh and skeleton
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    plt.title('Linear Skinning')

    colors_template = np.asarray([mcolors.to_rgba(c)
          for c in plt.rcParams['axes.prop_cycle'].by_key()['color']])

    # Draw mesh (points and edges)
    x, y = zip(*mesh.vertices)
    point_colors = np.ones((len(mesh.vertices), 4))

    num_bones = len(skeleton.bones)
    num_vertices = len(mesh.vertices)
    for vertex_id in range(num_vertices):
        point_color = np.zeros(3)

        for bone_id in range(num_bones):
            weight = weights[bone_id][vertex_id]
            point_color += colors_template[bone_id][0:3] * weight

        point_colors[vertex_id][0:3] = point_color

    ax.scatter(x, y, color=point_colors, s=3.0)

    segments = mesh.get_boundary_segments()
    line_segments = LineCollection(segments,
                               linewidths=1.0,
                               colors='orange',
                               linestyles='-',
                               alpha=1.0)
    ax.add_collection(line_segments)

    # Draw displacement
    if displacement is not None:
        segments = []
        for vertex_id, vertex in enumerate(mesh.vertices):
            segments.append([vertex, vertex+displacement[vertex_id]])

        line_segments = LineCollection(segments,
                               linewidths=1.0,
                               colors='green',
                               linestyles='-',
                               alpha=1.0)

        ax.add_collection(line_segments)



    # Draw skeleton
    segments = skeleton.get_bone_segments()
    line_segments = LineCollection(segments,
                                   linewidths=3.0,
                                   colors=colors_template,
                                   linestyles='-',
                                   alpha=1.0)

    ax.add_collection(line_segments)
    plt.show()

    # Export figure into a png file
    if len(render_folder_path) > 0:
        filename = str(frame_id).zfill(4) + " .png"
        fig.savefig(render_folder_path + "/" + filename)
