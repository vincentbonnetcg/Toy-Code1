"""
@author: Vincent Bonnet
@description : rod simulation with backward euler integrator
Implicit formulation and Conjugate Gradient (WIP)
"""

import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

'''
 Global Constants
'''
ROD_ROOT_POS = (0., 0.) # in meters
ROD_NUM_SEGMENTS = 4
ROD_LENGTH = 2.0 # in meters
ROD_STIFFNESS = 1.0 # in newtons per meter (N/m)
ROD_DAMPING = 0.01
ROD_PARTICLE_MASS = 0.001 # in Kg
GRAVITY = (0.0, -9.81) # in meters per second^2
NUM_FRAME = 100;
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 4 # number of substep per frame

'''
 Force Functions (Stretch and Damping)
'''
def stretchForce(x0, x1, rest, stiffness):
    direction = x0 - x1
    stretch = np.linalg.norm(direction)
    if (not np.isclose(stretch, 0.0)):
         direction /= stretch

    return direction * ((stretch - rest) * stiffness)

def dampingForce(x0, x1, v0, v1, damping):
    direction = x0 - x1
    stretch = np.linalg.norm(direction)
    if (not np.isclose(stretch, 0.0)):
        direction /= stretch
    relativeVelocity = v0 - v1
    return direction * (np.dot(relativeVelocity, direction) * damping)

'''
 Constraint Class
 Describes a constraint function and its first (gradient) and second (hessian) derivatives
'''
# Base Class Constraint
class BaseConstraint:
    def __init__(self, stiffness, damping, ids):
        self.stiffness = stiffness
        self.damping = damping
        self.ids = ids
        self.f = np.zeros((len(ids), 2)) # TODO - no need to store that - should be not be allocated in Baseconstraint
        self.dfdx = np.zeros((len(ids),2,2)) # TODO - no need to store that - should be not be allocated in Baseconstraint

    def applyForces(self, data):
        for i in range(len(self.ids)):
            data.f[self.ids[i]] += self.f[i]

    def computeForces(self, data):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeForces'")

    def computeJacobians(self, data):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeJacobians'")

    def getJacobian(self, data, fi, xj):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'getJacobian'")

# Constraint between particle and static point
class AnchorSpringConstraint(BaseConstraint):
    def __init__(self, stiffness, damping, ids, targetPos, data):
       BaseConstraint.__init__(self, stiffness, damping, ids)
       self.restLength = np.linalg.norm(targetPos - data.x[self.ids[0]])
       self.targetPos = targetPos

    def computeForces(self, data):
        x = data.x[self.ids[0]]
        v = data.v[self.ids[0]]
        force = stretchForce(x, self.targetPos, self.restLength, self.stiffness)
        force += dampingForce(x, self.targetPos, v, (0,0), self.damping)
        self.f[0] = force * -1
    
    def computeJacobians(self, data):
        x = data.x[self.ids[0]]
        self.dfdx[0] = stretchNumericalJacobiandf0dx0(x,  self.targetPos, self.restLength, self.stiffness) * -1
        
    def getJacobian(self, data, fi, xj):
        return self.dfdx[0]

# Constraint between two particles
class SpringConstraint(BaseConstraint):
    def __init__(self, stiffness, damping, ids, data):
        BaseConstraint.__init__(self, stiffness, damping, ids)
        self.restLength = np.linalg.norm(data.x[ids[0]] - data.x[ids[1]])

    def computeForces(self, data):
        x0 = data.x[self.ids[0]]
        x1 = data.x[self.ids[1]]
        v0 = data.v[self.ids[0]]
        v1 = data.v[self.ids[1]]
        force = stretchForce(x0, x1, self.restLength, self.stiffness)
        force += dampingForce(x0, x1, v0, v1, self.damping)
        self.f[0] = force * -1
        self.f[1] = force

    def computeJacobians(self, data):
        x0 = data.x[self.ids[0]]
        x1 = data.x[self.ids[1]]
        self.dfdx[1] = stretchNumericalJacobiandf0dx0(x0, x1, self.restLength, self.stiffness)
        self.dfdx[0] = self.dfdx[1] * -1

    def getJacobian(self, data, fi, xj):
        #(df/dx)ji = (df/dx)ij = Jx 
        #(df/dx)ii = (df/dx)jj = -Jx
        if (fi == xj):
            return self.dfdx[0]
        return self.dfdx[1]

'''
 Chain Class
'''
class Chain:
    def __init__(self, root, length, numEdges, particleMass, stiffness, damping):
        # Parameters
        self.numEdges = numEdges
        self.numVertices = numEdges + 1
        # Initialize particle state
        self.x = np.ones((self.numVertices, 2)) # position
        self.v = np.zeros((self.numVertices, 2)) # velocity
        self.m = np.ones(self.numVertices) * particleMass# mass
        self.im = 1.0 / self.m # inverse mass
        self.f = np.zeros((self.numVertices, 2)) #  force
        self.dfdx = np.zeros((self.numVertices,2,2)) # jacobian  # TODO - not really good !
        
        # Set position : start the rod in a horizontal position
        axisx = np.linspace(root[0], root[0]+length, num=self.numVertices, endpoint=True)
        for i in range(self.numVertices):
            self.x[i] = (axisx[i], root[1])

        # Initialize constraints
        self.constraints = []
        self.constraints.append(AnchorSpringConstraint(stiffness, damping, [0], root, self))
        for i in range(self.numEdges):
            self.constraints.append(SpringConstraint(stiffness, damping, [i, i+1], self))


'''
 Jacobian Forces
  |dfx/dx   dfx/dy|
  |dfy/dx   dfy/dy|
'''
def stretchNumericalJacobiandf0dx0(x0, x1, rest, stiffness):
    stencilSize = 0.0001 # stencil size for the central difference
    jacobian = np.zeros(shape=(2,2))
    
    # Derivative respective to x
    rx0 = np.add(x0, [stencilSize, 0])
    lx0 = np.add(x0, [-stencilSize, 0])
    rforce0 = stretchForce(rx0, x1, rest, stiffness)
    lforce0 = stretchForce(lx0, x1, rest, stiffness)
    gradientX = (rforce0 - lforce0) / (stencilSize * 2.0)
    
    # Derivative respective to y
    bx0 = np.add(x0, [0, -stencilSize])
    tx0 = np.add(x0, [0, stencilSize])
    bforce0 = stretchForce(bx0, x1, rest, stiffness)
    tforce0 = stretchForce(tx0, x1, rest, stiffness)
    gradientY = (tforce0 - bforce0) / (stencilSize * 2.0)
    
    # Set jacobian with gradients
    jacobian[0, 0] = gradientX[0]
    jacobian[1, 0] = gradientX[1]
    jacobian[0, 1] = gradientY[0]
    jacobian[1, 1] = gradientY[1]
    
    return jacobian

'''
 Implicit Step
 Solve : 
     (M - h^2 * df/dx) * deltaV = h * (fo + h * df/dx * v0)
       A = (M - h^2 * df/dx)
       b = h * (fo + h * df/dx * v0)
     => A * deltaV = b <=> deltaV = A^-1 * b    
     deltaX = (v0 + deltaV) * h
     v = v + deltaV 
     x = x + deltaX
'''
def implicitStep(chain, dt, gravity):   
    chain.f.fill(0.0)
    chain.dfdx.fill(0.0)
    # Add gravity
    for i in range(chain.numVertices):
        chain.f[i] += np.multiply(gravity, chain.m[i])

    # Prepare forces and jacobians
    for constraint in chain.constraints:
        constraint.computeForces(chain)
        constraint.computeJacobians(chain)
        constraint.applyForces(chain)

    # Assemble the system (Ax=b) where x is the change of velocity
    # Assemble A = (M - h^2 * df/dx)
    A = np.zeros((chain.numVertices * 2, chain.numVertices * 2))
    for i in range(chain.numVertices):
        massMatrix = np.matlib.identity(2) * chain.m[i]
        A[i*2:i*2+2,i*2:i*2+2] = massMatrix
    
    dfdxMatrix = np.zeros((chain.numVertices * 2, chain.numVertices * 2))
    for constraint in chain.constraints:
        ids = constraint.ids
        for fi in range(len(constraint.ids)):
            for xj in range(len(constraint.ids)):
                Jx = constraint.getJacobian(chain, fi, xj)
                dfdxMatrix[ids[fi]*2:ids[fi]*2+2,ids[xj]*2:ids[xj]*2+2] -= (Jx * dt * dt)
        
    A += dfdxMatrix
    
    # Assemble b = h *( f0 + h * df/dx * v0)
    # (f0 * h) + (df/dx * v0 * h * h)
    b = np.zeros((chain.numVertices * 2, 1))
    for i in range(chain.numVertices):
        b[i*2:i*2+2] += (np.reshape(chain.f[i], (2,1)) * dt)
    
    for constraint in chain.constraints:
        ids = constraint.ids
        for xi in range(len(constraint.ids)):
            fi = xi
            Jx = constraint.getJacobian(chain, fi, xi)
            b[fi*2:fi*2+2] += np.reshape(np.matmul(chain.v[ids[xi]], Jx), (2,1)) * dt * dt

    # Solve the system (Ax=b)
    deltaVArray = np.linalg.solve(A, b)
       
    # Advect
    for i in range(chain.numVertices):
        deltaV = [float(deltaVArray[i*2]), float(deltaVArray[i*2+1])]
        deltaX = (chain.v[i] + deltaV) * dt
        chain.v[i] += deltaV
        chain.x[i] += deltaX

'''
 Semi Implicit Step
'''
def semiImplicitStep(chain, dt, gravity):
    chain.f.fill(0.0)
    # Add gravity
    for i in range(chain.numVertices):
        chain.f[i] += np.multiply(gravity, chain.m[i])

    # Compute and add internal forces
    for constraint in chain.constraints:
        constraint.computeForces(chain)
        constraint.applyForces(chain)

    # Integrator
    for i in range(chain.numVertices):
        chain.v[i] += chain.f[i] * chain.im[i] * dt
        chain.x[i] += chain.v[i] * dt

'''
 Draw
'''
def draw(chain, frameId):
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
        }

    fig = plt.figure()
    plt.xlabel('x (in meters)')
    plt.ylabel('y (in meters)')
    plt.title('Mass-spring-damper - frame ' + str(frameId), fontdict=font)

    # Draw segments
    Path = mpath.Path
    pathVerts = chain.x
    pathCodes = [Path.MOVETO]
    
    for i in range(chain.numEdges): 
        pathCodes.append(Path.LINETO)
    
    path = Path(pathVerts, pathCodes)
    
    ax = fig.add_subplot(111)
    patch = mpatches.PathPatch(path, lw=1, fill=False)
    ax.add_patch(patch)
    ax.axis('equal')
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)

    # Draw vertices
    x, y = zip(*path.vertices)
    line, = ax.plot(x, y, 'go')
    
    plt.show()

'''
 Execute
'''
chain = Chain(ROD_ROOT_POS, ROD_LENGTH, ROD_NUM_SEGMENTS, ROD_PARTICLE_MASS, ROD_STIFFNESS, ROD_DAMPING)
draw(chain, 0)

for frameId in range(1, NUM_FRAME+1): 
    print("")
    dt = FRAME_TIMESTEP / NUM_SUBSTEP
    for substepId in range(NUM_SUBSTEP): 
        semiImplicitStep(chain, dt, GRAVITY)
        #implicitStep(chain, dt, GRAVITY)
    draw(chain, frameId)
