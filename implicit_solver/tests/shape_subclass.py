"""
@author: Vincent Bonnet
@description : Subclasses of the Shape class
"""

from core import Shape
import numpy as np

class WireShape(Shape):
    '''
    Creates a wire shape
    '''
    def __init__(self, startPoint, endPoint, num_edges):
        Shape.__init__(self, num_edges+1, num_edges, 0)

        axisx = np.linspace(startPoint[0], endPoint[0], num=num_edges+1, endpoint=True)
        axisy = np.linspace(startPoint[1], endPoint[1], num=num_edges+1, endpoint=True)

        for i in range(num_edges+1):
            self.vertex.position[i] = (axisx[i], axisy[i])

        # Set Edge Indices
        vertex_indices = []
        for i in range(num_edges):
            vertex_indices.append((i, i+1))

        self.edge.vertex_ids = np.array(vertex_indices, dtype=int)

class RectangleShape(Shape):
    '''
    Creates a rectangle
    '''
    def __init__(self, min_x, min_y, max_x, max_y):
        Shape.__init__(self, num_vertices=4, num_edges=5, num_faces=2)
        # Set positions
        self.vertex.position[0] = (min_x, min_y)
        self.vertex.position[1] = (min_x, max_y)
        self.vertex.position[2] = (max_x, max_y)
        self.vertex.position[3] = (max_x, min_y)
        # Set edges
        self.edge.vertex_ids[0] = (0, 1)
        self.edge.vertex_ids[1] = (1, 2)
        self.edge.vertex_ids[2] = (2, 0)
        self.edge.vertex_ids[3] = (2, 3)
        self.edge.vertex_ids[4] = (3, 0)
        # Set faces
        self.face.vertex_ids[0] = (0, 1, 2)
        self.face.vertex_ids[1] = (0, 2, 3)

class BeamShape(Shape):
    '''
    Creates a beam shape
    '''
    def __init__(self, position, width, height, cell_x, cell_y):
        Shape.__init__(self, (cell_x+1)*(cell_y+1))

        # Set Vertex position
        # 8 .. 9 .. 10 .. 11
        # 4 .. 5 .. 6  .. 7
        # 0 .. 1 .. 2  .. 3
        vertex_id = 0
        axisx = np.linspace(position[0], position[0]+width, num=cell_x+1, endpoint=True)
        axisy = np.linspace(position[1], position[1]+height, num=cell_y+1, endpoint=True)

        for j in range(cell_y+1):
            for i in range(cell_x+1):
                self.vertex.position[vertex_id] = (axisx[i], axisy[j])
                vertex_id += 1

        # Lambda function to get particle indices from cell coordinates
        cell_to_pids = lambda i, j: [i + (j*(cell_x+1)),
                                     i + (j*(cell_x+1)) + 1,
                                     i + ((j+1)*(cell_x+1)),
                                     i + ((j+1)*(cell_x+1)) + 1]

        # Set Edge Indices
        vertex_indices = []
        for j in range(cell_y):
            for i in range(cell_x):
                pids = cell_to_pids(i, j)

                vertex_indices.append((pids[1], pids[3]))
                if i == 0:
                    vertex_indices.append((pids[0], pids[2]))

                vertex_indices.append((pids[2], pids[3]))
                if j == 0:
                    vertex_indices.append((pids[0], pids[1]))


        self.edge.vertex_ids = np.array(vertex_indices, dtype=int)

        # Set Face Indices
        face_indices = []
        for j in range(cell_y):
            for i in range(cell_x):
                pids = cell_to_pids(i, j)

                face_indices.append((pids[0], pids[1], pids[2]))
                face_indices.append((pids[1], pids[2], pids[3]))

        self.face.vertex_ids = np.array(face_indices, dtype=int)
