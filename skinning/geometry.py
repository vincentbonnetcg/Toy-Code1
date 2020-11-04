"""
@author: Vincent Bonnet
@description : Geometry
"""

import numpy as np

def create_beam_mesh(min_x, min_y, max_x, max_y, cell_x, cell_y):
    '''
    Creates a mesh as a beam
    Example of beam with cell_x(3) and cell_y(2):
        |8 .. 9 .. 10 .. 11
        |4 .. 5 .. 6  .. 7
        |0 .. 1 .. 2  .. 3
    '''
    num_vertices = (cell_x + 1) * (cell_y + 1)
    vertices = np.zeros((num_vertices,2))

    # Set Points
    vertex_id = 0
    axisx = np.linspace(min_x, max_x, num=cell_x+1, endpoint=True)
    axisy = np.linspace(min_y, max_y, num=cell_y+1, endpoint=True)

    for j in range(cell_y+1):
        for i in range(cell_x+1):
            vertices[vertex_id] = (axisx[i], axisy[j])
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

    indices = np.array(edge_indices, dtype=int)

    return Mesh(vertices, indices)

class Mesh:
    '''
    Mesh contains a vertex buffer, index buffer and weights map for binding
    '''
    def __init__(self, vertices, indices):
        self.vertices = vertices
        self.indices = indices

    def get_boundary_segments(self):
        segments = []

        for vertex_ids in self.indices:
            segments.append([self.vertices[vertex_ids[0]],
                             self.vertices[vertex_ids[1]]])

        return segments
