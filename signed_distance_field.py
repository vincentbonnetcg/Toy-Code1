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

# Parameters
VOXEL_SIZE = 0.1


def ij_to_centroid(i, j, voxel_size):
    '''
    Voxel coordinates to world space
    '''
    return (i * voxel_size, j * voxel_size)

def discretize_polyline(polyline, voxel_size):
    '''
    Discretize polyline into a dense regular grid
    '''
    min_pos = np.min(polyline, axis=0) / voxel_size
    max_pos = np.max(polyline, axis=0) / voxel_size
    min_cell = np.floor(min_pos)
    max_cell = np.ceil(max_pos)
    num_cell_x = int(max_cell[0] - min_cell[0] + 1)
    num_cell_y = int(max_cell[1] - min_cell[1] + 1)
    grid = np.zeros((num_cell_x, num_cell_y))

    return min_cell, grid


def grid_to_centroid(min_cell, grid, voxel_size):
    '''
    Convert grid to points
    '''
    num_cell_x, num_cell_y = grid.shape
    total_points = num_cell_x * num_cell_y
    points = np.zeros((total_points, 2))
    point_ids = 0
    np.zeros(num_cell_x * num_cell_y)
    for i in range(num_cell_x):
        for j in range(num_cell_y):
            pos = ij_to_centroid(i + min_cell[0], j + min_cell[1], voxel_size)
            points[point_ids][0] = pos[0]
            points[point_ids][1] = pos[1]
            point_ids += 1

    return points

# Create polyline data
star = mpath.Path.unit_regular_star(6)
polygon_star = star.to_polygons()
polyline = polygon_star[0]

# Discretize polyline
min_cell, grid = discretize_polyline(polyline, VOXEL_SIZE)
points = grid_to_centroid(min_cell, grid, VOXEL_SIZE)

# Display
ax = plt.subplot(111)
col = mcol.PathCollection([star], facecolor='blue')
ax.add_collection(col)
ax.axis('equal')
ax.set_title('Signed distance polygon')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
plt.tight_layout()
px, py = zip(*points)
ax.plot(px, py, '.', alpha=0.5, color='orange', markersize = 5.0)
plt.show()
