"""
@author: Vincent Bonnet
@description : Signed distance field computation solving Eikonal Equation
Placeholder with a brute force implementation
It will be replaced by Fast Marchin Method or Fast Sweeping Method
"""
import matplotlib.pyplot as plt
import matplotlib.collections as mcol
import matplotlib as mpl
import numpy as np
import matplotlib.path as mpath

VOXEL_SIZE = 0.01

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

def project_position_on_segment(p0, p1, p2):
    '''
    Project point on segment p1-p2
    '''
    d = p2 - p1
    d_norm = np.linalg.norm(d)
    d /= d_norm
    t = np.dot(p0 - p1, d)
    t = min(max(t, 0.0), d_norm)
    return p1 + d * t

def project_position_on_mesh(mesh_points, mesh_edge_indices, pos):
    '''
    Project position on mesh (mesh_points, mesh_edge_indices)
    '''
    result_distance2 = np.finfo(np.float64).max
    result_pos = pos

    for i in range(mesh_edge_indices.shape[0]):
        edge_vtx_id = mesh_edge_indices[i]
        p1 = mesh_points[edge_vtx_id[0]]
        p2 = mesh_points[edge_vtx_id[1]]

        project_pos = project_position_on_segment(pos, p1, p2)
        distance2 = np.dot(pos - project_pos, pos - project_pos)
        if (distance2 < result_distance2):
            result_distance2 = distance2
            result_pos = project_pos

    return result_pos, np.sqrt(result_distance2)

class Mesh:
    def __init__(self, polyline):
        self.points = np.asarray(polyline)
        num_points = self.points.shape[0]
        num_edge_indices = num_points - 1
        self.edge_indices = np.zeros((num_edge_indices, 2), dtype=np.int32)
        for edge_id in range(num_edge_indices):
            self.edge_indices[edge_id][0] = edge_id
            self.edge_indices[edge_id][1] = edge_id+1

    def get_segments(self):
        segs = []
        for edge_vtx_id in self.edge_indices:
            seg = []
            for vtx_id in edge_vtx_id:
                seg.append(self.points[vtx_id])
            segs.append(seg)
 
        return segs

    def project_position(self, pos):
        '''
        Returns the position projected on mesh
        '''
        return project_position_on_mesh(self.points, self.edge_indices, pos)

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

def compute_distance(mesh, grid):
    '''
    Compute the distance field with a brute force method
    '''
    for i in range(grid.size[0]):
        for j in range(grid.size[1]):
            #pos = grid.ij_to_ws(i, j)
            pass
    # TODO

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
#compute_sign(mesh, grid) - NOT IMPLEMENTED
#compute_distance(mesh, grid) - NOT IMPLEMENTED
points = grid_to_points(grid)
distances = np.zeros(points.shape[0])
colors = np.ones((points.shape[0], 4))

norm = mpl.colors.Normalize(vmin=-1, vmax=1)
color_mapper = mpl.cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("Greens"))
for idx, point in enumerate(points):
     pt, dst = mesh.project_position(point)
     colors[idx] = color_mapper.to_rgba(dst)

# Display
ax = plt.subplot(111)
line_collection = mcol.LineCollection(mesh.get_segments())

ax.axis('equal')
ax.set_title('Distance Field')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])

ax.add_collection(line_collection)
px, py = zip(*points)
ax.scatter(x=px, y=py, s=1.0, c=colors, alpha=0.5)


plt.tight_layout()
plt.show()
