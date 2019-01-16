"""
@author: Vincent Bonnet
@description : Python code to bridge Houdini Data to the solver
This code should be pasted inside a Houdini Python geometry node
"""
import numpy as np

# TODO - copy face indices
# TODO - multiple objects
# TODO - import hou and node

# Input Data
time = hou.time()
node = hou.pwd()
geo = node.geometry()

# copy data : Houdini to Numpy
# Position / Edge
points = geo.points()
num_vertices = len(points)
pos_array = np.zeros((num_vertices, 3), dtype=float)
for id, point in enumerate(points):
    point = point.position()
    pos_array[id] = [point[0],point[1],point[2]]

edges = geo.globEdges("*")
num_edges = len(edges)
edge_array = np.zeros((num_edges, 2), dtype=int)
for id, edge in enumerate(edges):
    points = edge.points()
    edge_array[id] = [points[0].number(), points[1].number()]

# TODO - replace with solver
pos_array += [0,1 * time,0]

# copy data : Numpy to Houdini
for id, point in enumerate(geo.points()):
    point.setPosition(pos_array[id])
