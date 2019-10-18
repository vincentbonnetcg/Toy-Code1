"""
@author: Vincent Bonnet
@description : This file contains serializer and deserializer of shapes
"""

import lib.common as common
import numpy as np

def write_shape_to_npz_file(filename, pos, edge_vtx_ids, face_vtx_ids):
    np.savez(filename, positions = pos, edge_vertex_ids = edge_vtx_ids, face_vertex_ids = face_vtx_ids)

def create_shape_from_npz_file(filename):
    npzfile = np.load(filename)
    positions = npzfile['positions']
    edge_vertex_ids = npzfile['edge_vertex_ids']
    face_vertex_ids = npzfile['face_vertex_ids']
    num_vertices = len(positions)
    num_edges = len(edge_vertex_ids)
    num_faces = len(face_vertex_ids)

    shape = common.Shape(num_vertices, num_edges, num_faces)

    np.copyto(shape.vertex.position, positions)
    np.copyto(shape.edge.vertex_ids, edge_vertex_ids)
    np.copyto(shape.face.vertex_ids, face_vertex_ids)

    return shape
