"""
@author: Vincent Bonnet
@description : solve inverse kinematics problem with pseudo-inverse
"""

import numpy as np
from render_helper import RenderHelper

'''
 Global Constants
'''
NUM_SEGMENTS = 10
INITIAL_ANGLE_RANGE = 0.0 #  [-angle, angle] in degrees for initial segment angle
INITIAL_SEGMENT_LENGTH = 1000.0 # length each segment
TARGET_POS = (-500.0, 2000.0)

JACOBIAN_METHOD = "analytic" #  analytic / numerical
NUM_ITERATIONS = 50
MAX_STEP_SIZE = 150
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
 RenderChain Class
'''
class RenderChain(RenderHelper):
    def __init__(self, min_x, max_x, min_y, max_y):
        RenderHelper.__init__(self, min_x, max_x, min_y, max_y)
        
    def draw(self, data):
        pos = computePositions(chain)
        x, y = zip(*pos)
        
        # draw line, vertices and target
        self.ax.plot(x, y, "-.", markersize=2)
        self.ax.plot(x, y, 'go')
        self.ax.plot(TARGET_POS[0], TARGET_POS[1], 'ro')

'''
 Get positions from chain
'''
def computePositions(chain):
    rootPos = [0., 0.]
    totalPos = rootPos
    totalAngle = 0.
    
    positions = []
    positions.append(np.copy(rootPos))
    
    for i in range(chain.numSegments):
        totalAngle += chain.angles[i]
        x = np.cos(np.deg2rad(totalAngle + 90)) * chain.lengths[i]
        y = np.sin(np.deg2rad(totalAngle + 90)) * chain.lengths[i]
        totalPos[0] += x
        totalPos[1] += y
        positions.append(np.copy(totalPos))

    return positions

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
        vec = np.subtract(positions[chain.numSegments],  positions[i])
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
    vec = np.subtract(TARGET_POS, positions[chain.numSegments])
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
    diff = np.subtract(TARGET_POS, positions[chain.numSegments])
    diffNorm = np.linalg.norm(diff)
    if (diffNorm <= THRESHOLD):
        return True
    return False

'''
 Execute
'''
# prepare a chain
render = RenderChain(-5000., 5000., 0., 10000.)
chain = Chain()
forwardKinematic(chain, 1.0) 
# run inverse kinematic algorithm until convergence
iterations = 1
while iterations <= NUM_ITERATIONS and not hasReachTarget(chain):
    inverseKinematic(chain)
    print("IK : Iteration", iterations, "/", NUM_ITERATIONS )  
    render.prepare_figure()
    render.show_figure(chain)
    iterations += 1
