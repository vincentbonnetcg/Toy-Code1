"""
@author: Vincent Bonnet
@description : interpolation of 1D function with Radial Basis Functions
"""

import numpy as np
import matplotlib.pyplot as plt

'''
 Global Parameters
'''
def FUNCTION_1D(x):
    return np.sin(x) * np.cos((x+1)*2) * 2
MIN_RANGE = 0.0
MAX_RANGE = 10.0
NUM_SAMPLES = 15

'''
 Create random point from the polygon
'''
def randomSampleFromFunction1D(function, minRange, maxRange, numSamples):
    result = []
    samples_x = np.linspace(minRange, maxRange, numSamples, endpoint=True)
    
    for i in range(0,numSamples):
        result.append((samples_x[i], function(samples_x[i])))

    return result

'''
 RBF's kernels
  .r parameter is the euclidean distance
  .kernelParameter is currently hardcoded
 Multiquadric kernel is not used because it is not positive defined
'''
kernelParameter = 1.5
def gaussianKernel(r):
    return np.exp(-np.square((r * kernelParameter)))

def inverseQuadraticKernel(r):
    return 1.0 / (1.0 + np.square(r * kernelParameter))

def inverseMultiQuadraticKernel(r):
    return 1.0 / np.sqrt(1.0 + np.square(r * kernelParameter))

'''
 we should solve the system below
 Aw = b where
 
 b are the sampled points [y0, y1, y2]
 
 A is the interpolation matrix :
  | k(x0-x0) k(x0-x1) k(x0-x2) ... |
  | k(x1-x0) k(x1-x1) k(x1-x2) ... |
  | k(x2-x0) k(x2-x1) k(x2-x2) ... |
  |   ...      ...      ...    ... |
  
 w are the weights [w0, w1, w2] - unknown 
'''
def computeRBF_weights(points, kernel):
    numSamples = np.size(points,0)
    interpolationMatrix = np.zeros(shape=(numSamples,numSamples))
    for i in range(numSamples):
        for j in range(numSamples):
            interpolationMatrix[i,j] = kernel(points[j][0]-points[i][0])
    
    samplePoints = np.zeros(shape=(numSamples, 1))
    for i in range(numSamples):
        samplePoints[i]  = points[i][1]

    inverseInterpolationMatrix = np.linalg.inv(interpolationMatrix)
    weights = np.matmul(inverseInterpolationMatrix, samplePoints)
    
    return weights

'''
 Evaluate the RBF
'''
def radialBasisFunction(points, rfbWeights, kernel, x):
    numSamples = np.size(points,0)
    result = 0.0
    for i in range(numSamples):
        result += kernel(x - points[i][0]) * rfbWeights[i]
        
    return result

'''
 Drawing Methods
'''
def drawFunction1D(function1D, minRange, maxRange, step):
    # prepare figure
    fig, ax = plt.subplots()
    ax.grid()
    ax.axis('equal')
    # draw
    t = np.linspace(minRange, maxRange, int((maxRange - minRange) / step), endpoint=True)
    plt.plot(t, function1D(t), '-.', color="blue")
    # display
    plt.tight_layout()
    plt.show()

def drawRBF_1D(points, weights, kernel, referenceFunction, minRange, maxRange, step):   
    # prepare figure
    fig, ax = plt.subplots()
    ax.axis('equal')
    # draw reference function 
    t = np.linspace(minRange, maxRange, int((maxRange - minRange) / step), endpoint=True)
    plt.plot(t, referenceFunction(t), linestyle='solid', color="green", label="reference function")
    # draw rbf
    t = np.linspace(minRange, maxRange, int((maxRange - minRange) / step), endpoint=True)
    plt.plot(t, radialBasisFunction(points, weights, kernel, t), linestyle='dotted', color="blue", label="rbf interpolation")
    # draw points
    x, y = zip(*points)
    plt.plot(x, y, '*', color='red', label="samples")   
    # display
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16 }
    plt.title('RBF interpolation', fontdict=font)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(bbox_to_anchor=(1, 1), loc=2)
    plt.show()  

'''
 Execute
'''
# sample the 1D function and compute the RBF weights
points = randomSampleFromFunction1D(FUNCTION_1D, MIN_RANGE, MAX_RANGE, NUM_SAMPLES)
weights = computeRBF_weights(points, gaussianKernel)

# display RBF
drawRBF_1D(points, weights, gaussianKernel, FUNCTION_1D, MIN_RANGE,MAX_RANGE, 0.1)

# Debugging of the RBF kernels
#drawFunction1D(gaussianKernel, -4.0, 4.0, 0.05)
#drawFunction1D(inverseQuadraticKernel, -4.0, 4.0, 0.05)
#drawFunction1D(inverseMultiQuadraticKernel, -4.0, 4.0, 0.05)

