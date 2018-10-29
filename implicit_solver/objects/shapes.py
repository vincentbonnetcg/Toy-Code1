"""
@author: Vincent Bonnet
@description : Shape description
"""

import numpy as np

class ShapeVertexComponent:
    '''
    Vertex Component
    '''
    def __init__(self, num_vertices):
        self.position = np.zeros((num_vertices, 2), dtype=float)

    def size(self):
        return len(self.position)

    def __str__(self):
        return str(self.__class__)  + ": " + str(self.__dict__)

class ShapeEdgeComponent:
    '''
    Edge Component
    '''
    def __init__(self, num_edges):
        self.vertex_ids = np.zeros((num_edges, 2), dtype=int)

    def size(self):
        return len(self.vertex_ids)

    def __str__(self):
        return str(self.__class__)

class ShapeFaceComponent:
    '''
    Face Component
    '''
    def __init__(self, num_faces):
        self.vertex_ids = np.zeros((num_faces, 3), dtype=int)

    def size(self):
        return len(self.vertex_ids)

    def __str__(self):
        return str(self.__class__)


class Shape:
    '''
    Shape Description
    '''
    def __init__(self, num_vertices, num_edges = 0, num_faces = 0):
        self.vertex = ShapeVertexComponent(num_vertices)
        self.edge = ShapeEdgeComponent(num_edges)
        self.face = ShapeFaceComponent(num_faces)

    def num_vertices(self):
        return self.vertex.size()

    def num_edges(self):
        return self.edge.size()

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

        # TODO - need to add the connectivities

class BeamShape(Shape):
    '''
    Creates a wire shape
    '''
    def __init__(self, position, width, height, cell_x, cell_y):
        Shape.__init__(self, (cell_x+1)*(cell_y+1))

        # Set Vertex position
        # 8 .. 9 .. 10 .. 11
        # 4 .. 5 .. 6  .. 7
        # 0 .. 1 .. 2  .. 3
        self.cell_x = cell_x
        self.cell_y = cell_y
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
        for j in range(self.cell_y):
            for i in range(self.cell_x):
                pids = cell_to_pids(i, j)

                face_indices.append((pids[0], pids[1], pids[2]))
                face_indices.append((pids[1], pids[2], pids[3]))

        self.face.vertex_ids = np.array(face_indices, dtype=int)
