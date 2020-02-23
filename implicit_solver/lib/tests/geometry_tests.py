"""
@author: Vincent Bonnet
@description : Unit tests for geometry functions
"""

import unittest
import lib.common as common

'''
Tests for geometry functions
'''
def createTriangulatedSquareShape():
    shape = common.Shape(num_vertices=4, num_edges=5, num_faces=2)
    shape.vertex[:] = ((-1.0, -1.0),(-1.0, 1.0),(1.0, 1.0),(1.0, -1.0))
    shape.edge[:] = ((0, 1),(1,2),(2, 0),(2, 3),(3,0))
    shape.face[:] = ((0, 1, 2),(0, 2, 3))
    return shape

class Tests(unittest.TestCase):
    def test_edges_on_surface(self):
        shape = createTriangulatedSquareShape()
        edges_ids, edge_normals = shape.get_edge_surface_data()
        self.assertEqual(len(edges_ids), 4)
        self.assertEqual(len(edge_normals), 4)

    def setUp(self):
        print(" Geometry Test:", self._testMethodName)

if __name__ == '__main__':
    unittest.main(Tests())
