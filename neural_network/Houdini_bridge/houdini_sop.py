"""
@author: Vincent Bonnet
@description : Python code to bridge Houdini Data to the solver
This code should be pasted inside a Houdini Python geometry node
"""

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

    # Collect Position
    points = geo.points()
    num_vertices = len(points)
    pos_array = np.zeros((num_vertices, 3), dtype=float)
    for i, point in enumerate(points):
        point = point.position()
        pos_array[i] = [point[0],point[1],point[2]]

    # Collect Edges
    edges = geo.globEdges("*")
    num_edges = len(edges)
    edge_array = np.zeros((num_edges, 2), dtype=int)
    for i, edge in enumerate(edges):
        points = edge.points()
        edge_array[i] = [points[0].number(), points[1].number()]

    # Collect Polygon (Triangles)
    primitives = geo.prims()
    num_triangles = len(primitives)
    triangle_array = np.zeros((num_triangles, 3), dtype=int)
    for i, primitive in enumerate(primitives):
        points = primitive.points()
        if len(points) == 3:
            triangle_array[i] = [points[0].number(), points[1].number(), points[2].number()]

    return pos_array, edge_array, triangle_array

def store_mesh_data(folder, pos_array, edge_array, tri_array):
    if len(folder) > 0:
        np.savetxt(folder+'/pos.txt', pos_array)
        np.savetxt(folder+'/edge.txt', edge_array)
        np.savetxt(folder+'/tri.txt', tri_array)

def set_position_from_folder(geo, folder):
    if len(folder) > 0:
        pos_array = np.loadtxt(folder+'/pos.txt')
        for id, point in enumerate(geo.points()):
            point.setPosition(pos_array[id])

# Input Data
time = hou.time()
node = hou.pwd()
geo = node.geometry()
folder = '' # e.g c:/folder/folder2

# Export Geo
pos_array, edge_array, tri_array = fetch_mesh_data(geo)
#store_mesh_data(folder, pos_array, edge_array, tri_array)

# Import Geo Position
set_position_from_folder(geo, folder)



