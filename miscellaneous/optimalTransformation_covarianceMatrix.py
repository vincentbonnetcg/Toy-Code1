"""
@author: Vincent Bonnet
@description : find optimal rigid transformation by using Principal component analysis
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

'''
 Transform point
'''
def transformPoints(pointData, angle, scale, translate):
    cosAngle = np.cos(np.deg2rad(angle))
    sinAngle = np.sin(np.deg2rad(angle))

    # create scale-rotation matrix
    scaleRotationMat = np.matrix([[cosAngle * scale[0],  -sinAngle * scale[1]],
                                  [sinAngle * scale[0],  cosAngle * scale[1]]])

    # Non vectorized code
    #for i in range(NUM_POINTS):
    #    pointData[i] = np.matmul(scaleRotationMat, pointData[i])
    #    pointData[i] += translate

    transformedPoint = np.matmul(scaleRotationMat, pointData.T)
    np.copyto(pointData, transformedPoint.T)
    pointData += translate

'''
 Utility functions
'''
def computeCentroid(pointData):
    return np.sum(pointData, axis=0) / np.size(pointData,0)

def covariance(pointArray, index0, index1):
    # Non vectorized code
    #cov = 0
    #for p in pointArray:
    #    cov += p[index0] * p[index1]

    cov = np.sum(pointArray[:,index0] * pointArray[:,index1])

    return cov / np.size(pointArray,0)

def computeBestRotation(pointData, centroid):
    # get direction
    localPoint = np.copy(pointData)
    np.subtract(localPoint, centroid, out=localPoint)
    # compute covariance matrix
    # cov(X,X) cov(X,Y)
    # cov(Y,X) cov(Y,Y)
    covXX = covariance(localPoint,0,0)
    covYY = covariance(localPoint,1,1)
    covXY = covariance(localPoint,0,1)
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

def draw(pointData, centroid, rotationMatrix):
    '''
     Draw point, frames, box
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)

    transformedPointData = np.copy(pointData)
    transformedPointData -= centroid
    inverseRotationMatrix = rotationMatrix.T
    transformedPointData = np.asarray(np.matmul(inverseRotationMatrix, transformedPointData.T).T)
    boxMin = np.min(transformedPointData, axis=0)
    boxMax = np.max(transformedPointData, axis=0)

    # draw the local frame
    # in ax.arrow(...)  vec.y and vec.x are reversed !
    vec0x = rotationMatrix.item(0,0)
    vec0y = rotationMatrix.item(0,1)
    vec1x = rotationMatrix.item(1,0)
    vec1y = rotationMatrix.item(1,1)
    ax.arrow(centroid[0], centroid[1], vec0y, vec0x, head_width=1.0, facecolor='red', edgecolor='black')
    ax.arrow(centroid[0], centroid[1], vec1y, vec1x, head_width=1.0, facecolor='green', edgecolor='black')

    # draw points
    x, y = zip(*pointData)
    ax.scatter(x, y,s=0.5)

    # draw box
    vertex = np.asarray([(boxMin[0], boxMin[1]), (boxMin[0], boxMax[1]),
                         (boxMax[0], boxMax[1]), (boxMax[0], boxMin[1])])
    vertex = np.asarray(np.matmul(rotationMatrix, vertex.T).T)
    vertex += centroid
    polygon = patches.Polygon(vertex, closed=True,ec='darkgreen',lw=2,fill=False)
    ax.add_patch(polygon)

    # show result
    plt.show()

if __name__ == '__main__':
    pointData = np.array(np.random.rand(NUM_POINTS, 2) - 0.5)  # positions [-0.5, -0.5] [0.5, 0.5]
    transformPoints(pointData, ROTATION, SCALE, TRANSLATE)

    # compute best transform (rotation, centroid)
    centroid = computeCentroid(pointData)
    rotationMatrix = computeBestRotation(pointData, centroid)

    draw(pointData, centroid, rotationMatrix)
