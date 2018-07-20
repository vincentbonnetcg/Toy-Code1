"""
@author: Vincent Bonnet
@description : rod simulation with backward euler integrator
Implicit formulation and Conjugate Gradient (WIP)
"""

import constraints as cn
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
NUM_FRAME = 10;
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 4 # number of substep per frame

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
        
        # Set position : start the rod in a horizontal position
        axisx = np.linspace(root[0], root[0]+length, num=self.numVertices, endpoint=True)
        for i in range(self.numVertices):
            self.x[i] = (axisx[i], root[1])

        # Initialize constraints
        self.constraints = []
        self.constraints.append(cn.AnchorSpringConstraint(stiffness, damping, [0], root, self))
        for i in range(self.numEdges):
            self.constraints.append(cn.SpringConstraint(stiffness, damping, [i, i+1], self))

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
