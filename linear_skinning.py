"""
@author: Vincent Bonnet
@description : Linear Skinning
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

class Bone:
    def __init__(self, length = 1.0, rotation = 0.0):
        self.length = length
        self.rotation = rotation # in degrees
        self.bone_parent = None
        self.bone_children = []

    def get_homogenous_transform(self):
        '''
        3x3 Matrix combining rotation and displacement
        where R is 2x2 rotation matrix
        and d is 2d vector
        | R  d |
        | 0  1 |
        '''
        H = np.zeros((3,3))
        cos = np.cos(np.deg2rad(self.rotation))
        sin = np.sin(np.deg2rad(self.rotation))
        H[0, 0] = cos
        H[1, 1] = cos
        H[0, 1] = -sin
        H[1, 0] = sin
        H[0, 2] = cos * self.length
        H[1, 2] = sin * self.length
        H[2, 2] = 1.0
        return H

class Skeleton:
    def __init__(self, root_position, root_bone):
        self.root_position = np.asarray(root_position)
        self.root_bone = root_bone
        self.bones = []

    def add_bone(self, bone, bone_parent = None):
        self.bones.append(bone)
        if bone_parent is not None:
            bone.bone_parent = bone_parent
            bone_parent.bone_children.append(bone)

    def get_bone_segments(self):
        segments = []

        bone = self.root_bone
        prev_H = np.identity(3)
        prev_H[0, 2] = self.root_position[0]
        prev_H[1, 2] = self.root_position[1]
        H = np.copy(prev_H)
        while bone is not None:
            prev_H = H
            bone_H = bone.get_homogenous_transform()
            H = np.matmul(H, bone_H)
            if len(bone.bone_children) > 0:
                bone = bone.bone_children[0]
            else:
                bone = None

            homogenous_coordinate = np.asarray([0.0, 0.0, 1.0])
            start_pos = np.matmul(prev_H, homogenous_coordinate)
            end_pos = np.matmul(H, homogenous_coordinate)
            segments.append([start_pos[0:2], end_pos[0:2]])

        return segments

class Mesh:
    def __init__(self, vertex_buffer, index_buffer):
        self.vertex_buffer = np.asarray(vertex_buffer)
        self.index_buffer = np.asarray(index_buffer)

    def get_boundary_segments(self):
        segments = []

        for vertex_ids in self.index_buffer:
            segments.append([self.vertex_buffer[vertex_ids[0]],
                             self.vertex_buffer[vertex_ids[1]]])

        return segments


def create_beam_mesh(min_x, min_y, max_x, max_y, cell_x, cell_y):
    '''
    Creates a beam by returning a vertex and index buffer
    Example :
        |8 .. 9 .. 10 .. 11
        |4 .. 5 .. 6  .. 7
        |0 .. 1 .. 2  .. 3
    '''
    num_vertices = (cell_x + 1) * (cell_y + 1)
    vertex_buffer = np.zeros((num_vertices,2))

    # Set Points
    vertex_id = 0
    axisx = np.linspace(min_x, max_x, num=cell_x+1, endpoint=True)
    axisy = np.linspace(min_y, max_y, num=cell_y+1, endpoint=True)

    for j in range(cell_y+1):
        for i in range(cell_x+1):
            vertex_buffer[vertex_id] = (axisx[i], axisy[j])
            vertex_id += 1

    # Set Edge Indices
    cell_to_ids = lambda i, j: i + (j*(cell_x+1))
    edge_indices = []
    for j in range(cell_y):
        ids = [cell_to_ids(0, j), cell_to_ids(0, j+1)]
        edge_indices.append(ids)
        ids = [cell_to_ids(cell_x, j), cell_to_ids(cell_x, j+1)]
        edge_indices.append(ids)

    for i in range(cell_x):
        ids = [cell_to_ids(i, 0), cell_to_ids(i+1, 0)]
        edge_indices.append(ids)
        ids = [cell_to_ids(i, cell_y), cell_to_ids(i+1, cell_y)]
        edge_indices.append(ids)

    index_buffer = np.array(edge_indices, dtype=int)

    return Mesh(vertex_buffer, index_buffer)

def create_skeleton():
    root_bone = Bone(length = 3.0, rotation = 5.0)
    bone1 = Bone(length = 3.0, rotation = -5.0)
    bone2 = Bone(length = 3.0, rotation = 5.0)
    bone3 = Bone(length = 3.0, rotation = -5.0)

    skeleton = Skeleton([-6.0, 0.0], root_bone)
    skeleton.add_bone(root_bone)
    skeleton.add_bone(bone1, root_bone)
    skeleton.add_bone(bone2, bone1)
    skeleton.add_bone(bone3, bone2)

    return skeleton

def draw(mesh, skeleton):
    fig = plt.figure()
    font = {'color':  'darkblue',
                 'weight': 'normal',
                 'size': 18}
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.title('Linear Skinning', fontdict = font)

    # Draw mesh (points and edges)
    x, y = zip(*mesh.vertex_buffer)
    ax.plot(x, y, '.', alpha=1.0, color='blue', markersize = 2.0)

    segments = mesh.get_boundary_segments()
    line_segments = LineCollection(segments,
                               linewidths=2.0,
                               colors='orange',
                               linestyles='-',
                               alpha=0.5)
    ax.add_collection(line_segments)

    # Draw skeleton
    colors = [mcolors.to_rgba(c)
              for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    segments = skeleton.get_bone_segments()
    line_segments = LineCollection(segments,
                                   linewidths=3.0,
                                   colors=colors,
                                   linestyles='-',
                                   alpha=1.0)

    ax.add_collection(line_segments)
    plt.show()

def main():
    mesh = create_beam_mesh(-7.0, -1.5, 7.0, 1.5, 10, 6)
    skeleton = create_skeleton()
    draw(mesh, skeleton)

if __name__ == '__main__':
    main()


