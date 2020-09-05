"""
@author: Vincent Bonnet
@description : Graph optimization (Greedy colouring algorithm)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as collections

def get_mesh(filename):
    npzfile = np.load(filename)
    vertices = npzfile['positions']
    edge_ids = npzfile['edge_vertex_ids']
    face_ids = npzfile['face_vertex_ids']
    return vertices, edge_ids, face_ids

def compute_groups(mesh):
    adjencies = get_adjacencies(mesh)
    vertices = mesh[0]
    num_vertices = len(vertices)
    group_ids = [-1] * num_vertices

    for node, adjacencies in adjencies.items():
        # get group ids from adjacencies
        adjacency_groups = []
        for adj in adjacencies:
            adjacency_groups.append(group_ids[adj])

        # search for unassigned group id
        group_id = 0
        while group_id in adjacency_groups:
            group_id += 1

        group_ids[node] = group_id

    return np.asarray(group_ids)

def get_adjacencies(mesh):
    adjacencies = {}
    edge_ids = mesh[1]

    for ids in edge_ids:
        for i in range(2):
            neighbors = adjacencies.get(ids[i], [])
            neighbors.append(ids[(i+1)%2])
            adjacencies[ids[i]] = neighbors

    return adjacencies


def show(mesh):
    vertices = mesh[0]
    face_ids = mesh[2]
    group_ids = compute_groups(mesh)

    # display the graph
    # Only support up to 20 difference colours (see cmap=plt.cm.tab20)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis('equal')
    min_range = np.min(vertices, axis=0)
    max_range = np.max(vertices, axis=0)
    ax.set_xlim(min_range[0], max_range[0])
    ax.set_ylim(min_range[1], max_range[1])

    font = {'family': 'serif',
            'color':  'darkblue',
            'weight': 'normal',
            'size': 14 }
    num_groups = np.max(group_ids)
    plt.title(('Greedy Coloring Algorithm (%d colors)'%num_groups), fontdict=font)
    plt.axis('off')

    triangles = []
    for face_id in face_ids:
        v0 = vertices[face_id[0]]
        v1 = vertices[face_id[1]]
        v2 = vertices[face_id[2]]
        triangles.append([v0, v1, v2])

    # draw mesh
    collec = collections.PolyCollection(triangles, facecolors='white',
                                                    edgecolors='black',
                                                    linewidths=0.1)
    ax.add_collection(collec)

    # draw nodes
    colors = ['blue', 'red', 'yellow', 'green', 'orange', 'pink']
    for group_id in range(num_groups):
        node_indices, = np.where(group_ids == group_id)
        if len(node_indices)==0:
            continue

        vtx = vertices[node_indices]
        x, y = zip(*vtx)
        ax.plot(x, y, '.', alpha=1.0, color=colors[group_id], markersize = 5.0)

    plt.show()

if __name__ == '__main__':
    vertices, edge_ids, face_ids = get_mesh('rabbit.npz')
    mesh = (vertices, edge_ids, face_ids)
    show(mesh)
