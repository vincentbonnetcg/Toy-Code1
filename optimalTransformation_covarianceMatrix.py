"""
@author: Vincent Bonnet
@description : find optimal rigid transformation by using Principal component analysis
"""

import numpy as np
import matplotlib.pyplot as plt

'''
 Global Constants
'''
NUM_POINTS = 1000
ROTATION = 39.0
SCALE = [12.0, 4.0]
TRANSLATE = [2.5, 1]

'''
 Point Data
'''
class PointData:
        def __init__(self):
            self.x = np.array(np.random.rand(NUM_POINTS, 2) - 0.5)  # positions [-0.5, -0.5] [0.5, 0.5]

''' 
 Transform point
'''
def transformPoints(pointData, angle, scale, translate):
    cosAngle = np.cos(np.deg2rad(angle))
    sinAngle = np.sin(np.deg2rad(angle))
    
    # create scale-rotation matrix
    scaleRotationMax = np.matrix([[cosAngle * scale[0],  -sinAngle * scale[1]],
                                  [sinAngle * scale[0],  cosAngle * scale[1]]])
    
    # translate points
    for i in range(NUM_POINTS): 
        vec = np.matmul(scaleRotationMax, pointData.x[i])
        vec += translate
        pointData.x[i]= vec

'''
 Utility functions
''' 
def numElements(pointData):
    return np.size(pointData.x,0)

def computeCentroid(pointData):
    return np.sum(pointData.x, axis=0) / numElements(pointData)

def covariance(pointArray, index0, index1):
    cov = 0
    for p in pointArray:
        cov += p[index0] * p[index1]
    return cov / (np.size(pointArray,0))

def computeOrthogonalFrame(pointData, centroid):
    # get direction
    localPoint = np.copy(pointData.x)  
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
    return v

'''
 Draw point data and frame
'''
def drawPointsAndFrame(pointData):
    # create figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    
    # compute eigenvectors
    centroid = computeCentroid(pointData)
    frames = computeOrthogonalFrame(pointData, centroid)
    
    vec0 = np.reshape(frames[0], (2, 1))
    vec1 = np.reshape(frames[1], (2, 1))
    vec0x = np.asscalar(vec0[0])
    vec0y = np.asscalar(vec0[1])
    vec1x = np.asscalar(vec1[0])
    vec1y = np.asscalar(vec1[1])
    
    # draw the local frame
    # in ax.arrow(...)  vec.y and vec.x are reversed !
    ax.arrow(centroid[0], centroid[1], vec0y, vec0x, head_width=1.0, facecolor='red', edgecolor='black')
    ax.arrow(centroid[0], centroid[1], vec1y, vec1x, head_width=1.0, facecolor='green', edgecolor='black')
    
    # draw points
    x, y = zip(*pointData.x)
    ax.scatter(x, y,s=0.5)
    
    # show result
    plt.show()

'''
 Execute
'''
pointData = PointData()
transformPoints(pointData, ROTATION, SCALE, TRANSLATE)

drawPointsAndFrame(pointData)
