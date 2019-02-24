"""
@author: Vincent Bonnet
@description : Shape description
"""

import numpy as np

class VertexComponent:
    '''
    Vertex Component
    '''
    def __init__(self, num_vertices):
        self.position = np.zeros((num_vertices, 2), dtype=float)

    def size(self):
        return len(self.position)

    def __str__(self):
        return str(self.__class__)  + ": " + str(self.__dict__)

class EdgeComponent:
    '''
    Edge Component
    '''
    def __init__(self, num_edges):
        self.vertex_ids = np.zeros((num_edges, 2), dtype=int)

    def size(self):
        return len(self.vertex_ids)

    def vertex_edges_dict(self):
        return component_to_vertex_ids(self.vertex_ids)

    def __str__(self):
        return str(self.__class__)

class FaceComponent:
    '''
    Face Component
    '''
    def __init__(self, num_faces):
        self.vertex_ids = np.zeros((num_faces, 3), dtype=int)

    def size(self):
        return len(self.vertex_ids)

    def __str__(self):
        return str(self.__class__)

def component_to_vertex_ids(vertex_ids):
    result = {}
    for component_id, vertex_ids in enumerate(vertex_ids):
        for vertex_id in vertex_ids:
            result.setdefault(vertex_id, []).append(component_id)

    return result

def vertex_ids_neighbours(vertex_ids):
    result = {}
    for component_id, vertex_ids in enumerate(vertex_ids):
        for vertex_id0 in vertex_ids:
            for vertex_id1 in vertex_ids:
                if vertex_id0 != vertex_id1:
                    result.setdefault(vertex_id0, []).append(vertex_id1)
    return result

class Shape:
    '''
    Shape Description
    '''
    def __init__(self, num_vertices, num_edges=0, num_faces=0):
        self.vertex = VertexComponent(num_vertices)
        self.edge = EdgeComponent(num_edges)
        self.face = FaceComponent(num_faces)

    def extract_transform_from_shape(self):
        '''
        Returns the 'optimal' position and modify the shape vertices from world space to local space
        Optimal rotation is not computed
        '''
        optimal_pos = np.average(self.vertex.position, axis=0)
        np.subtract(self.vertex.position, optimal_pos, out=self.vertex.position)
        optimal_rot = 0
        return optimal_pos, optimal_rot

    def num_vertices(self):
        return self.vertex.size()

    def num_edges(self):
        return self.edge.size()

    def num_faces(self):
        return self.face.size()

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
