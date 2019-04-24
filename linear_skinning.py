"""
@author: Vincent Bonnet
@description : Linear Skinning (Skeletal Subspace Deformation)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors

def distance_from_segment(p, seg_p1, seg_p2):
    '''
    Distance from segment [seg_p1, seg_p2] and point p
    '''
    d = seg_p2 - seg_p1
    d_norm = np.linalg.norm(d)
    d_normalized = d / d_norm
    t = np.dot(p - seg_p1, d_normalized)
    t = min(max(t, 0.0), d_norm)
    projected_p = seg_p1 + d_normalized * t
    return np.linalg.norm(p - projected_p)

class Bone:
    def __init__(self, length = 1.0, rotation = 0.0):
        self.length = length
        self.rotation = rotation # in degrees
        # hirerarchy info
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

    def attach_mesh(self, mesh, max_influences, kernel_func):
        num_vertices = len(mesh.vertex_buffer)
        num_bones = len(self.bones)
        mesh.weights_map = np.zeros((num_bones, num_vertices))
        bone_segments = self.get_bone_segments()

        # compute weights per bone per vertices (weights map)
        for vertex_id, vertex in enumerate(mesh.vertex_buffer):
            for bone_id, bone_seg in enumerate (bone_segments):
                distance = distance_from_segment(vertex, bone_seg[0], bone_seg[1])
                mesh.weights_map[bone_id][vertex_id] = kernel_func(distance)

        # updates the weights map by limiting ...
        # the number of influences by picking the n closest vertices
        num_influences = min(num_bones, max_influences)
        for vertex_id, vertex in enumerate(mesh.vertex_buffer):
            vertex_weights = np.zeros(num_bones)
            for bone_id, bone_seg in enumerate (bone_segments):
                vertex_weights[bone_id] = mesh.weights_map[bone_id][vertex_id]


            vertex_weigths_sorted_index = np.argsort(vertex_weights)
            for vtx_id in range(num_bones - num_influences):
                vertex_weights[vertex_weigths_sorted_index[vtx_id]] = 0.0

            vertex_weights /= np.sum(vertex_weights)
            for bone_id, bone_seg in enumerate (bone_segments):
                mesh.weights_map[bone_id][vertex_id] = vertex_weights[bone_id]

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
    '''
    Mesh contains a vertex buffer, index buffer and weights map for binding
    '''
    def __init__(self, vertex_buffer, index_buffer):
        self.vertex_buffer = np.asarray(vertex_buffer)
        self.index_buffer = np.asarray(index_buffer)
        self.weights_map = None # influence for each bones

    def get_boundary_segments(self):
        segments = []

        for vertex_ids in self.index_buffer:
            segments.append([self.vertex_buffer[vertex_ids[0]],
                             self.vertex_buffer[vertex_ids[1]]])

        return segments

def create_beam_mesh(min_x, min_y, max_x, max_y, cell_x, cell_y):
    '''
    Creates a mesh as a beam
    Example of beam with cell_x(3) and cell_y(2):
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
    '''
    Create a skeleton object
    '''
    root_bone = Bone(length = 3.0, rotation = 1.0)
    bone1 = Bone(length = 3.0, rotation = -1.0)
    bone2 = Bone(length = 3.0, rotation = 1.0)
    bone3 = Bone(length = 3.0, rotation = -1.0)

    skeleton = Skeleton([-6.0, 0.0], root_bone)
    skeleton.add_bone(root_bone)
    skeleton.add_bone(bone1, root_bone)
    skeleton.add_bone(bone2, bone1)
    skeleton.add_bone(bone3, bone2)

    return skeleton

def draw(mesh, skeleton):
    '''
    Drawing function to display the mesh and skeleton
    '''
    fig = plt.figure()
    font = {'color':  'darkblue',
                 'weight': 'normal',
                 'size': 18}
    ax = fig.add_subplot(111)
    ax.axis('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.title('Linear Skinning', fontdict = font)

    colors_template = [mcolors.to_rgba(c)
          for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

    # Draw mesh (points and edges)
    x, y = zip(*mesh.vertex_buffer)
    point_colors = np.ones((len(mesh.vertex_buffer), 4))

    num_bones = len(skeleton.bones)
    num_vertices = len(mesh.vertex_buffer)
    for vertex_id in range(num_vertices):
        point_color = np.zeros(3)

        for bone_id in range(num_bones):
            weight = mesh.weights_map[bone_id][vertex_id]
            point_color += (np.asarray(colors_template[bone_id])[0:3] * weight)

        point_colors[vertex_id][0:3] = point_color

    ax.scatter(x, y, color=point_colors, s=10.0)

    segments = mesh.get_boundary_segments()
    line_segments = LineCollection(segments,
                               linewidths=1.0,
                               colors='orange',
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

def main():
    '''
    Main
    '''
    mesh = create_beam_mesh(-7.0, -1.5, 7.0, 1.5, 15, 3)
    skeleton = create_skeleton()

    kernal_parameter = 1.0
    kernel_function = lambda v : np.exp(-np.square((v * kernal_parameter)))

    skeleton.attach_mesh(mesh, max_influences = 2, kernel_func = kernel_function)
    draw(mesh, skeleton)

if __name__ == '__main__':
    main()


