"""
@author: Vincent Bonnet
@description : find optimal rigid transformation by using Principal Component Analysis
Using eigen decomposition or svd decomposition
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

NUM_POINTS = 1000
ROTATION = 39.0
SCALE = [11.0, 4.0]
TRANSLATE = [2.5, 1]

def transformPoints(points):
    cosAngle = np.cos(np.deg2rad(ROTATION))
    sinAngle = np.sin(np.deg2rad(ROTATION))
    mat = np.asarray([[cosAngle * SCALE[0],  -sinAngle * SCALE[1]],
                      [sinAngle * SCALE[0],  cosAngle * SCALE[1]]])
    np.copyto(points, np.matmul(mat, points.T).T)
    points += TRANSLATE

def covariance(points, index0, index1):
    return np.sum(points[:,index0] * points[:,index1]) / len(points)

def computeBestRotation(local_points):
    # compute covariance matrix
    # cov(X,X) cov(X,Y)
    # cov(Y,X) cov(Y,Y)
    covXX = covariance(local_points,0,0)
    covYY = covariance(local_points,1,1)
    covXY = covariance(local_points,0,1)
    covariance_matrix = np.asarray([[covXX, covXY],
                                  [covXY, covYY]])
    # covariance decomposition
    w, v = np.linalg.eig(covariance_matrix)
    rotation_matrix = v
    det = np.linalg.det(rotation_matrix)
    if det < 0.0:
        # from reflection matrix to rotation matrix
        rotation_matrix *= -1.0
    return rotation_matrix

def draw(points, centroid, rotation_matrix):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    # compute min and max
    inverse_matrix = rotation_matrix.T
    transformed_points = np.copy(points)
    transformed_points -= centroid
    transformed_points = np.matmul(inverse_matrix, transformed_points.T).T
    box_min = np.min(transformed_points, axis=0)
    box_max = np.max(transformed_points, axis=0)

    # draw the local frame
    eigvec0 = rotation_matrix[:,0]
    eigvec1 = rotation_matrix[:,1]
    ax.arrow(centroid[0], centroid[1], eigvec0[0], eigvec0[1], head_width=1.0, facecolor='red', edgecolor='black')
    ax.arrow(centroid[0], centroid[1], eigvec1[0], eigvec1[1], head_width=1.0, facecolor='green', edgecolor='black')

    # draw points
    x, y = zip(*points)
    ax.scatter(x, y,s=0.5)

    # draw box
    vertex = np.asarray([(box_min[0], box_min[1]), (box_min[0], box_max[1]),
                         (box_max[0], box_max[1]), (box_max[0], box_min[1])])

    vertex = np.matmul(rotation_matrix, vertex.T).T
    vertex += centroid
    polygon = patches.Polygon(vertex, closed=True,ec='darkgreen',lw=2,fill=False)
    ax.add_patch(polygon)

    # show result
    plt.show()

if __name__ == '__main__':
    # create point cloud
    points = np.random.rand(NUM_POINTS, 2) - 0.5
    transformPoints(points)

    # compute best transform (centroid + matrix)
    centroid = np.mean(points, axis=0)
    local_points = points - centroid
    rotation_matrix = computeBestRotation(local_points)

    # draw result
    draw(points, centroid, rotation_matrix)
