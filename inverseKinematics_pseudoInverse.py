"""
@author: Vincent Bonnet
@description : solve inverse kinematics problem with pseudo-inverse
"""

import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

'''
 Global Constants
'''
NUM_SEGMENTS = 10
INITIAL_ANGLE_RANGE = 0.0 #  [-angle, angle] in degrees for initial segment angle
INITIAL_SEGMENT_LENGTH = 1000.0 # length each segment
TARGET_POS = (0.0, 0.0)

JACOBIAN_METHOD = "analytic" #  analytic / numerical
NUM_ITERATIONS = 200
MAX_STEP_SIZE = 100
THRESHOLD = 1.0 # acceptable distance between the last vertex and target position

'''
 Chain Class
'''
class Chain:
    def __init__(self):
        # up is the vector (0,1)
        self.angles = (np.random.rand(NUM_SEGMENTS) * 2.0 - 1) * INITIAL_ANGLE_RANGE
        self.lengths = np.ones(NUM_SEGMENTS) * INITIAL_SEGMENT_LENGTH
        self.numSegments = NUM_SEGMENTS
        
'''
 Get positions from chain
'''
def computePositions(chain):
    rootPos = (0., 0.)
    totalPos = rootPos
    totalAngle = 0.
    
    positions = [rootPos]
    
    for i in range(chain.numSegments):
        totalAngle += chain.angles[i]
        x = np.cos(np.deg2rad(totalAngle + 90)) * chain.lengths[i]
        y = np.sin(np.deg2rad(totalAngle + 90)) * chain.lengths[i]      
        totalPos = np.add(totalPos, (x, y))
        positions.append(totalPos)

    return positions

'''
 Draw Chain and final target position
'''
def draw(chain, targetPosition):
    Path = mpath.Path
    
    # draw segments   
    pathVerts = computePositions(chain)
    pathCodes = [Path.MOVETO]
    
    for i in range(chain.numSegments): 
        pathCodes.append(Path.LINETO)
      
    path = Path(pathVerts, pathCodes)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = mpatches.PathPatch(path, facecolor='orange', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(-5000,5000)
    ax.set_ylim(0,10000)
       
    # draw vertices
    x, y = zip(*path.vertices)
    line, = ax.plot(x, y, 'go')
    
    # draw target
    line, = ax.plot(targetPosition[0], targetPosition[1], 'ro')
    
    plt.show()

'''
# Jacobian Helper functions
Layout of the partial differential matrix of size(m x n)
'm' is the number of DOF for the end effector (only translate XY)
'n' is the number of joints multiplied by the number of joint DOF
    joint0      joint1     joint2
    [angle0]   [angle1]   [angle2]
x :  dx/da0     dx/da1     dx/da2
y :  dy/da0     dy/da1     dy/da2
'''
#  Use central difference to approximate the differentiation
def computeNumericalJacobian(chain):
    jacobian = np.zeros(shape=(2,chain.numSegments))
    angleDt = 0.01
    for i in range(chain.numSegments):
        keepAngle = chain.angles[i]
        # compute the derivative
        chain.angles[i] = keepAngle + (angleDt * 0.5)
        forwardPositions = computePositions(chain)
        chain.angles[i] = keepAngle - (angleDt * 0.5)
        backwardPositions = computePositions(chain)
        dpda = (forwardPositions[chain.numSegments] - backwardPositions[chain.numSegments]) / angleDt
        jacobian[0, i] = dpda[0]
        jacobian[1, i] = dpda[1]
        # resort angle
        chain.angles[i] = keepAngle
    return np.matrix(jacobian)

def computeAnalyticJacobian(chain):
    positions = computePositions(chain)
    jacobian = np.zeros(shape=(2,chain.numSegments))
    for i in range(chain.numSegments):
        vec = positions[chain.numSegments] - positions[i]
        x = vec[0]
        y = vec[1]
        # compute the derivative
        # sin(np.deg2rad(1.0)) is the angular velocity
        # x = norm(vec) * cos(angle) and x' = norm(vec) * -sin(angle)
        # y = norm(vec) * sin(angle) and y' = norm(vec) * cos(angle)
        # hence x' = y * -1 and y' = x
        jacobian[0, i] = y * -1 * np.sin(np.deg2rad(1.0))
        jacobian[1, i] = x * np.sin(np.deg2rad(1.0))
    
    return np.matrix(jacobian)

def computePseudoInverse(chain):
    if (JACOBIAN_METHOD=="analytic"):
        jacobian = computeAnalyticJacobian(chain)
    else:
        jacobian = computeNumericalJacobian(chain)
    
    # pseudo-inverse from numpy to validate our implementation below
    #return np.linalg.pinv(jacobian) 
    
    # use own pseudo-inverse
    jacobiantInv = jacobian * jacobian.transpose()
    jacobiantInv = np.linalg.inv(jacobiantInv)
    return(jacobian.transpose() * jacobiantInv)

'''
 Forward Kinematics
 This is an unpretentious function to animate forward
'''
def forwardKinematic(chain, angle):
    for i in range(chain.numSegments):
        chain.angles[i] += angle

'''
 Inverse Kinematics
'''
def inverseKinematic(chain):
    positions = computePositions(chain)
    vec = TARGET_POS - positions[chain.numSegments]
    vecNorm = np.linalg.norm(vec)
    if (vecNorm > MAX_STEP_SIZE):
        vec /= vecNorm
        vec *= MAX_STEP_SIZE

    pseudoInverse = computePseudoInverse(chain)
    deltaAngles = np.matmul(pseudoInverse, np.reshape(vec, (2,1)))
    
    for i in range(chain.numSegments):
        chain.angles[i] += deltaAngles[i]

'''
 Function to determinate whether or not an 'acceptable' solution has been reached
''' 
def hasReachTarget(chain):
    positions = computePositions(chain)
    diff = TARGET_POS - positions[chain.numSegments]
    diffNorm = np.linalg.norm(diff)
    if (diffNorm <= THRESHOLD):
        return True
    return False

'''
 Execute
'''
# prepare a chain
chain = Chain()
forwardKinematic(chain, 1.0) 
# run inverse kinematic algorithm
iterations = 1
while iterations <= NUM_ITERATIONS and not hasReachTarget(chain):
    inverseKinematic(chain)
    print("IK : Iteration", iterations, "/", NUM_ITERATIONS )  
    draw(chain, TARGET_POS) 
    iterations += 1
