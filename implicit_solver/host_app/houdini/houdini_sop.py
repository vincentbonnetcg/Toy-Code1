"""
@author: Vincent Bonnet
@description : Python code to bridge Houdini Data to the solver
This code should be pasted inside a Houdini Python geometry node
"""

import numpy as np

def fetch_mesh_data(geo):
    pos_array = None
    edge_array = None
    triangle_array = None

    # get points and primitives
    points = geo.points()
    edges = geo.globEdges("*")
    primitives = geo.prims()

    # allocate numpy arrays
    num_vertices = len(points)
    num_edges = len(edges)
    num_triangles = len(primitives)
    pos_array = np.zeros((num_vertices, 2), dtype=float)
    edge_array = np.zeros((num_edges, 2), dtype=int)
    triangle_array = np.zeros((num_triangles, 3), dtype=int)

    # Collect Position
    for i, point in enumerate(points):
        point = point.position()
        pos_array[i] = [point[0],point[1]]

    # Collect Edges
    for i, edge in enumerate(edges):
        points = edge.points()
        edge_array[i] = [points[0].number(), points[1].number()]

    # Collect Polygon (Triangles)
    for i, primitive in enumerate(primitives):
        points = primitive.points()
        if len(points) == 3:
            triangle_array[i] = [points[0].number(), points[1].number(), points[2].number()]

    return pos_array, edge_array, triangle_array

# from host_app.rpc.shape_io
def write_shape_to_npz_file(filename, pos, edge_vtx_ids, face_vtx_ids):
    np.savez(filename, positions = pos, edge_vertex_ids = edge_vtx_ids, face_vertex_ids = face_vtx_ids)


# Input Data
time = hou.time()
node = hou.pwd()
geo = node.geometry()
filename = '' # e.g c:/folder/folder2/file.npz

# Export Geo
pos_array, edge_array, tri_array = fetch_mesh_data(geo)
write_shape_to_npz_file(filename, pos_array, edge_array, tri_array)



