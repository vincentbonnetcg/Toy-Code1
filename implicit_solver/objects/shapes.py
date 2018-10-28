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
    Vertex Component
    '''
    def __init__(self, num_edges):
        self.vertex_ids = np.zeros((num_edges, 2), dtype=int)

    def size(self):
        return len(self.vertex_ids)

    def __str__(self):
        return str(self.__class__)

class Shape:
    '''
    Shape Description
    '''
    def __init__(self, num_vertices, num_edges):
        self.vertex = ShapeVertexComponent(num_vertices)
        self.edge = ShapeEdgeComponent(num_edges)

    def num_vertices(self):
        return self.vertex.size()

    def num_edges(self):
        return self.edge.size()

class WireShape(Shape):
    '''
    Creates a wire shape
    '''
    def __init__(self, startPoint, endPoint, num_edges):
        Shape.__init__(self, num_edges+1, num_edges)

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
        Shape.__init__(self, (cell_x+1)*(cell_y+1), 0)

        self.cell_x = cell_x
        self.cell_y = cell_y

        # Vertex position
        # 8 .. 9 .. 10 .. 11
        # 4 .. 5 .. 6  .. 7
        # 0 .. 1 .. 2  .. 3
        self.cell_x = cell_x
        self.cell_y = cell_y
        vertex_id = 0
        cell_width = width / cell_x
        cell_height = height / cell_y
        for j in range(cell_y+1):
            for i in range(cell_x+1):
                self.vertex.position[vertex_id] = (i * cell_width + position[0], j * cell_height + position[1])
                vertex_id += 1

        # TODO - need to add the connectivities
