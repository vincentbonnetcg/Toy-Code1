"""
@author: Vincent Bonnet
@description : Signed distance field computation solving Eikonal Equation
Placeholder with a brute force implementation
It will be replaced by Fast Marchin Method or Fast Sweeping Method
"""

import matplotlib.pyplot as plt
import matplotlib.collections as mcol
import numpy as np
import matplotlib.path as mpath
import numba
import math

VOXEL_SIZE = 0.1

star = mpath.Path.unit_regular_star(6)
polygon_star = star.to_polygons()
POLYLINE = polygon_star[0]

class Grid:
    def __init__(self, min_cell, max_cell, voxel_size):
        self.size = [int(max_cell[0] - min_cell[0] + 1) ,
                     int(max_cell[1] - min_cell[1] + 1)]
        self.array = np.zeros((self.size[0], self.size[1]))
        self.voxel_size = voxel_size
        self.min_cell = min_cell
        self.max_cell = max_cell

    def ij_to_ws(self, i, j):
        return ((i + self.min_cell[0]) * self.voxel_size,
                (j + self.min_cell[1]) * self.voxel_size)

class Mesh:
    def __init__(self, polyline):
        self.points = np.asarray(polyline)
        num_points = self.points.shape[0]
        num_edge_indices = num_points - 1
        self.edge_indices = np.zeros((num_edge_indices, 2))
        for edge_id in range(num_edge_indices):
            self.edge_indices[edge_id][0] = edge_id
            self.edge_indices[edge_id][1] = edge_id+1

def discretize_polyline(mesh, voxel_size, margin = 1):
    '''
    Discretize polyline into a dense regular grid
    '''
    min_pos = np.min(mesh.points, axis=0) / voxel_size
    max_pos = np.max(mesh.points, axis=0) / voxel_size
    min_cell = np.floor(min_pos) - margin
    max_cell = np.ceil(max_pos) + margin
    grid = Grid(min_cell, max_cell, voxel_size)

    return grid

def distance_from_line(p0, p1, p2):
    '''
    Compute the distance between p0 and the segment p1-p2
    '''
    d = p2 - p1
    # TODO
    pass

def compute_distance(mesh, grid):
    '''
    Compute the distance field with a brute force method
    '''
    for i in range(grid.size[0]):
        for j in range(grid.size[1]):
            pos = grid.ij_to_ws(i, j)
    # TODO
    pass

def compute_sign(mesh, grid):
    '''
    Compute the sign with crossing number algorithm
    '''
    # TODO
    pass

def grid_to_points(grid):
    '''
    Convert grid to points
    '''
    total_points = grid.size[0] * grid.size[1]
    points = np.zeros((total_points, 2))
    point_ids = 0
    np.zeros(grid.size[0] * grid.size[1])
    for i in range(grid.size[0]):
        for j in range(grid.size[1]):
            pos = grid.ij_to_ws(i, j)
            points[point_ids][0] = pos[0]
            points[point_ids][1] = pos[1]
            point_ids += 1

    return points

# Discretize polyline
mesh = Mesh(POLYLINE)
grid = discretize_polyline(mesh, VOXEL_SIZE)
compute_sign(mesh, grid)
compute_distance(mesh, grid)
points = grid_to_points(grid)

# Display
ax = plt.subplot(111)
col = mcol.PathCollection([star], facecolor='blue')
ax.add_collection(col)
ax.axis('equal')
ax.set_title('Signed distance polygon')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
plt.tight_layout()
# TODO - draw distance with a certain colour
px, py = zip(*points)
ax.plot(px, py, '.', alpha=0.5, color='orange', markersize = 5.0)
plt.show()
