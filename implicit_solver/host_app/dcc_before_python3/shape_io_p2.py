"""
@author: Vincent Bonnet
@description : This file contains serializer and deserializer of shapes
It is only uses because Houdini and Maya do not support Python3 yet.
This file will go away as soon as those two DCCs move to Python3
"""

import pickle
import core

def write_to_file(points, edge_ids, face_ids, filename):
    out_file = open(filename,'wb')
    out_dict = {'points' : points, 'edge_ids' : edge_ids, 'face_ids' : face_ids }
    pickle.dump(out_dict, out_file)
    out_file.close()

def read_from_file(filename):
    in_file = open(filename,'rb')
    in_dict = pickle.load(in_file)
    in_file.close()
    points = in_dict['points']
    edge_ids = in_dict['edge_ids']
    face_ids = in_dict['face_ids']
    return points, edge_ids, face_ids

def create_shape_from_file(filename):
    # Load Data from file
    in_file = open(filename,'rb')
    in_dict = pickle.load(in_file)
    in_file.close()
    points = in_dict['points']
    edge_ids = in_dict['edge_ids']
    face_ids = in_dict['face_ids']

    # Create shape
    num_vertices = len(points)
    num_edges = len(edge_ids)
    num_faces = len(face_ids)
    shape = core.Shape(num_vertices, num_edges, num_faces)

    for i in range(num_vertices):
        shape.vertex.position[i] = (points[i][0], points[i][1])

    for i in range(num_edges):
        shape.edge.vertex_ids[i] = (edge_ids[i][0], edge_ids[i][1])

    for i in range(num_faces):
        shape.face.vertex_ids[i] = (face_ids[i][0], face_ids[i][1], face_ids[i][2])

    return shape