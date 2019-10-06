"""
@author: Vincent Bonnet
@description : solve inverse kinematics problem with pseudo-inverse and damped least square
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

JACOBIAN_METHOD = 'analytic' #  analytic / numerical
INVERSE_METHOD = 'damped_least_square' #  pseudo_inverse / damped_least_square
DAMPING_CONSTANT = 1.0 # used when INVERSE_METHOD == 'damped_least_square'

NUM_ITERATIONS = 100
MAX_STEP_SIZE = 100
THRESHOLD = 1.0 # acceptable distance between the last vertex and target position

'''
 Chain Class
'''
class Chain:
    def __init__(self):
        self.angles = np.ones(NUM_SEGMENTS) * INITIAL_ANGLE_RANGE
        self.lengths = np.ones(NUM_SEGMENTS) * INITIAL_SEGMENT_LENGTH
        self.numSegments = NUM_SEGMENTS

    def compute_positions(self):
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
 RenderChain Class
'''
class RenderChain(RenderHelper):
    def __init__(self, min_x, max_x, min_y, max_y):
        RenderHelper.__init__(self, min_x, max_x, min_y, max_y)

    def draw(self, chain):
        pos = chain.compute_positions()
        x, y = zip(*pos)

        # draw chain
        self.ax.plot(x, y, "b-", markersize=2) # draw blue segments
        self.ax.plot(x, y, 'go') # draw green points
        self.ax.plot(TARGET_POS[0], TARGET_POS[1], 'ro') # draw red target

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
        forwardPositions = chain.compute_positions()
        chain.angles[i] = keepAngle - (angleDt * 0.5)
        backwardPositions = chain.compute_positions()
        dpda = (forwardPositions[chain.numSegments] - backwardPositions[chain.numSegments]) / angleDt
        jacobian[0, i] = dpda[0]
        jacobian[1, i] = dpda[1]
        # resort angle
        chain.angles[i] = keepAngle
    return np.matrix(jacobian)

def computeAnalyticJacobian(chain):
    positions = chain.compute_positions()
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

'''
Compute the Inverse of the Jacobians
'''
def computePseudoInverse(jacobian):
    # pseudo-inverse from numpy to validate our implementation below
    #return np.linalg.pinv(jacobian)

    jacobiantInv = jacobian * jacobian.transpose()
    jacobiantInv = np.linalg.inv(jacobiantInv)
    return(jacobian.transpose() * jacobiantInv)

def computeDampedLeastSquare(jacobian):
    damping_matrix_constant = np.identity(2) * DAMPING_CONSTANT

    jacobiantInv = jacobian * jacobian.transpose()
    jacobiantInv += damping_matrix_constant
    jacobiantInv = np.linalg.inv(jacobiantInv)
    return(jacobian.transpose() * jacobiantInv)

'''
Singular values analysis (Only for debugging)
'''
def print_singluar_values(matrix):
    '''
    Print the singular values to indicate whether the matrix inversion is stable
    Large singular values would make it unstable
    matrix is decompose into U.E.Vt
    where U and V are othogonal and E is the diagonal matrix containing singular values
    hence its inverse is V.1/E.Ut
    '''
    singular_values = np.linalg.svd(matrix, compute_uv = False)
    print("-- singular_values --")
    print(singular_values)

'''
 Inverse Kinematics
'''
def inverseKinematic(chain):
    positions = chain.compute_positions()
    vec = np.subtract(TARGET_POS, positions[chain.numSegments])
    vecNorm = np.linalg.norm(vec)
    if (vecNorm > MAX_STEP_SIZE):
        vec /= vecNorm
        vec *= MAX_STEP_SIZE

    if JACOBIAN_METHOD == 'analytic':
        jacobian = computeAnalyticJacobian(chain)
    else:
        jacobian = computeNumericalJacobian(chain)

    if INVERSE_METHOD == 'pseudo_inverse':
        pseudoInverse = computePseudoInverse(jacobian)
    else:
        pseudoInverse = computeDampedLeastSquare(jacobian)

    # Debugging
    #print_singluar_values(pseudoInverse)

    deltaAngles = np.matmul(pseudoInverse, np.reshape(vec, (2,1)))
    for i in range(chain.numSegments):
        chain.angles[i] += deltaAngles[i]

'''
 Function to determinate whether or not an 'acceptable' solution has been reached
'''
def hasReachTarget(chain):
    positions = chain.compute_positions()
    diff = np.subtract(TARGET_POS, positions[chain.numSegments])
    if (np.linalg.norm(diff) <= THRESHOLD):
        return True
    return False

'''
 Execute
'''
# prepare a chain
render = RenderChain(-5000., 5000., 0., 10000.)
chain = Chain()
render.prepare_figure()
render.show_figure(chain)
# run inverse kinematic algorithm until convergence
iterations = 1
while iterations <= NUM_ITERATIONS and not hasReachTarget(chain):
    inverseKinematic(chain)
    print("IK : Iteration", iterations, "/", NUM_ITERATIONS )
    render.prepare_figure()
    render.show_figure(chain)
    iterations += 1
