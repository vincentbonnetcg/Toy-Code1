"""
@author: Vincent Bonnet
@description : solve inverse kinematics problem with damped least square
"""

import numpy as np
import matplotlib.pyplot as plt

NUM_SEGMENTS = 10
TARGET_POS = np.asarray([-500.0, 2000.0])
REGULARIZATION_TERM = 0.1
NUM_ITERATIONS = 100
MAX_STEP_SIZE = 100
THRESHOLD = 1.0 # acceptable distance between the last vertex and target position

class Chain:
    def __init__(self):
        chain_length = 10000.0
        segment_length = chain_length / NUM_SEGMENTS
        self.angles = np.zeros(NUM_SEGMENTS)
        self.lengths = np.ones(NUM_SEGMENTS) * segment_length
        self.x = np.empty((NUM_SEGMENTS+1, 2)) # positions
        computePositions(self)

def computePositions(chain):
    chain.x[0] = [0, 0]
    angle = 90.
    for i in range(NUM_SEGMENTS):
        angle += chain.angles[i]
        chain.x[i+1][0] = np.cos(np.deg2rad(angle))
        chain.x[i+1][1] = np.sin(np.deg2rad(angle))
        chain.x[i+1] *= chain.lengths[i]
        chain.x[i+1] += chain.x[i]

'''
Layout of the partial differential matrix of size(m x n)
'm' is the number of DOF for the end effector (only translate XY)
'n' is the number of joints multiplied by the number of joint DOF
    joint0      joint1     joint2
    [angle0]   [angle1]   [angle2]
x :  dx/da0     dx/da1     dx/da2
y :  dy/da0     dy/da1     dy/da2
'''
def analyticJacobian(chain):
    J = np.zeros(shape=(2,NUM_SEGMENTS))
    for i in range(NUM_SEGMENTS):
        vec = chain.x[-1] -  chain.x[i]
        # x = norm(vec) * cos(angle) and x' = norm(vec) * -sin(angle)
        # y = norm(vec) * sin(angle) and y' = norm(vec) * cos(angle)
        # hence x' = y * -1 and y' = x
        J[0, i] = vec[1] * -1 * np.sin(np.deg2rad(1.0))
        J[1, i] = vec[0] * np.sin(np.deg2rad(1.0))

    return np.matrix(J)

'''
Compute the Inverse of the Jacobians
'''
def dampedLeastSquare(J):
    damping_matrix_constant = np.identity(2) * REGULARIZATION_TERM
    invJ = J * J.transpose()
    invJ += damping_matrix_constant
    invJ = np.linalg.inv(invJ)
    return(J.transpose() * invJ)

def printSingluarValues(matrix):
    '''
    Singular values analysis (for debugging)
    Print the singular values to indicate whether the matrix inversion is stable
    SVD decompose the matrix into U.E.Vt
    where U and V are othogonal and E is the diagonal matrix containing singular values
    hence its inverse is V.1/E.Ut
    Large singular values makes the matrix inverse going to infinity
    '''
    singular_values = np.linalg.svd(matrix, compute_uv = False)
    print("-- singular_values --")
    print(singular_values)


def inverseKinematic(chain):
    vec = np.reshape(TARGET_POS - chain.x[-1], (2,1))
    #vec = TARGET_POS - x[-1]
    vec_norm = np.linalg.norm(vec)
    if (vec_norm > MAX_STEP_SIZE):
        vec /= vec_norm
        vec *= MAX_STEP_SIZE

    J = analyticJacobian(chain)
    pseudoInverse = dampedLeastSquare(J)

    # Debugging
    #printSingluarValues(pseudoInverse)

    delta_angles = np.matmul(pseudoInverse, vec)
    for i in range(NUM_SEGMENTS):
        chain.angles[i] += delta_angles[i]

def hasReachTarget(chain):
    if (np.linalg.norm(TARGET_POS - chain.x[-1]) <= THRESHOLD):
        return True
    return False

def show(chain):
    x, y = zip(*chain.x)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    min_x, max_x = -5000., 5000.
    min_y, max_y = 0., 10500
    #ax.axis('equal') # FIXME - causes problem
    ax.autoscale(enable=False)
    ratio = fig.get_size_inches()[0] / fig.get_size_inches()[1]
    offset = ((max_x - min_x) * ratio - (max_y - min_y)) / 2
    ax.set_xlim(min_x-offset, max_x+offset)
    ax.set_ylim(min_y, max_y)
    # draw chain
    ax.plot(x, y, "b-", markersize=2) # draw blue segments
    ax.plot(x, y, 'go') # draw green points
    ax.plot(TARGET_POS[0], TARGET_POS[1], 'ro') # draw red target
    plt.show()

if __name__ == '__main__':
    chain = Chain()
    show(chain)
    iterations = 1
    while iterations <= NUM_ITERATIONS and not hasReachTarget(chain):
        inverseKinematic(chain)
        print("IK : Iteration", iterations, "/", NUM_ITERATIONS )
        iterations += 1
        computePositions(chain)
        show(chain)

