"""
@author: Vincent Bonnet
@description : find optimal rigid transformation by using Principal Component Analysis
Using eigen decomposition or svd decomposition
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''
 Global Constants
'''
NUM_POINTS = 1000
ROTATION = 39.0
SCALE = [11.0, 4.0]
TRANSLATE = [2.5, 1]

def transformPoints(points, angle, scale, translate):
    cosAngle = np.cos(np.deg2rad(angle))
    sinAngle = np.sin(np.deg2rad(angle))

    # create scale-rotation matrix
    scaleRotationMat = np.matrix([[cosAngle * scale[0],  -sinAngle * scale[1]],
                                  [sinAngle * scale[0],  cosAngle * scale[1]]])

    transformedPoint = np.matmul(scaleRotationMat, points.T)
    np.copyto(points, transformedPoint.T)
    points += translate

def covariance(points, index0, index1):
    return np.sum(points[:,index0] * points[:,index1]) / len(points)

def computeBestRotation(points, centroid):
    # get direction
    local_points = np.copy(points)
    local_points -= centroid
    # compute covariance matrix
    # cov(X,X) cov(X,Y)
    # cov(Y,X) cov(Y,Y)
    covXX = covariance(local_points,0,0)
    covYY = covariance(local_points,1,1)
    covXY = covariance(local_points,0,1)
    covarianceMatrix = np.matrix([[covXX, covXY],
                                  [covXY, covYY]])
    # covariance decomposition
    w, v = np.linalg.eig(covarianceMatrix)
    rotationMatrix = v
    det = np.linalg.det(rotationMatrix)
    if det < 0.0:
        # from reflection matrix to rotation matrix
        rotationMatrix *= -1.0
    return rotationMatrix

def draw(points, centroid, rotationMatrix):
    '''
     Draw points + frames + bounding box
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')

    transformed_points = np.copy(points)
    transformed_points -= centroid
    inverseRotationMatrix = rotationMatrix.T
    transformed_points = np.asarray(np.matmul(inverseRotationMatrix, transformed_points.T).T)
    boxMin = np.min(transformed_points, axis=0)
    boxMax = np.max(transformed_points, axis=0)

    # draw the local frame
    # in ax.arrow(...)  vec.y and vec.x are reversed !
    vec0x = rotationMatrix.item(0,0)
    vec0y = rotationMatrix.item(0,1)
    vec1x = rotationMatrix.item(1,0)
    vec1y = rotationMatrix.item(1,1)
    ax.arrow(centroid[0], centroid[1], vec0y, vec0x, head_width=1.0, facecolor='red', edgecolor='black')
    ax.arrow(centroid[0], centroid[1], vec1y, vec1x, head_width=1.0, facecolor='green', edgecolor='black')

    # draw points
    x, y = zip(*points)
    ax.scatter(x, y,s=0.5)

    # draw box
    vertex = np.asarray([(boxMin[0], boxMin[1]), (boxMin[0], boxMax[1]),
                         (boxMax[0], boxMax[1]), (boxMax[0], boxMin[1])])

    vertex = vertex = np.asarray(np.matmul(rotationMatrix, vertex.T).T)
    vertex += centroid
    polygon = patches.Polygon(vertex, closed=True,ec='darkgreen',lw=2,fill=False)
    ax.add_patch(polygon)

    # show result
    plt.show()

if __name__ == '__main__':
    points = np.array(np.random.rand(NUM_POINTS, 2) - 0.5)  # positions [-0.5, -0.5] [0.5, 0.5]
    transformPoints(points, ROTATION, SCALE, TRANSLATE)

    # compute best transform (rotation, centroid)
    centroid = np.mean(points, axis=0)
    rotationMatrix = computeBestRotation(points, centroid)

    draw(points, centroid, rotationMatrix)
