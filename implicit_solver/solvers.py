"""
@author: Vincent Bonnet
@description : symplectic and backward Euler integrators
"""

import numpy as np
import scipy.sparse as sparse
import profiler

'''
 Base Solver
'''
class BaseSolver:
    def __init__(self, dt, stepsPerFrame):
        self.dt = dt # delta time per step
        self.stepsPerFrame = stepsPerFrame # number of step per frame
        self.currentTime = 0.0

    @profiler.timeit
    def solveFrame(self, scene):
        for substepId in range(self.stepsPerFrame):
            self.currentTime += self.dt
            self.preStep(scene, self.currentTime)
            self.step(scene, self.dt)

    def preStep(self, scene, time):
        scene.updateKinematics(time)

    def step(self, scene, dt):
        self.prepareSystem(scene, dt)
        self.assembleSystem(scene, dt)
        self.solveSystem(scene, dt)

    def prepareSystem(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'prepareSystem'")

    def assembleSystem(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'assembleSystem'")

    def solveSystem(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'solveSystem'")


'''
 Implicit Step
 Solve : 
     (M - h * df/dv - h^2 * df/dx) * deltaV = h * (fo + h * df/dx * v0)
       A = (M - h^2 * df/dx)
       b = h * (fo + h * df/dx * v0)
     => A * deltaV = b <=> deltaV = A^-1 * b    
     deltaX = (v0 + deltaV) * h
     v = v + deltaV 
     x = x + deltaX
'''
class ImplicitSolver(BaseSolver):
    def __init__(self, dt, stepsPerFrame):
        BaseSolver.__init__(self, dt, stepsPerFrame)
        # used to store system Ax=b
        self.A = None
        self.b = None

    @profiler.timeit
    def prepareSystem(self, scene, dt):
        # Set gravity
        for dynamic in scene.dynamics:
            dynamic.f.fill(0.0)
            for i in range(dynamic.numParticles):
                dynamic.f[i] += np.multiply(scene.gravity, dynamic.m[i])

        # Prepare forces and jacobians
        constraintsIterator = scene.getConstraintsIterator()
        for constraint in constraintsIterator:
            constraint.computeForces(scene)
            constraint.computeJacobians(scene)
            constraint.applyForces(scene)

    @profiler.timeit
    def assembleSystem(self, scene, dt):
        # Assemble the system (Ax=b) where x is the change of velocity
        totalParticles = scene.numParticles()
        # attempt to create 'row-based linked list' sparse matrix for simple sparse matrix construction
        # with A = sparse.lil_matrix((totalParticles * 2, totalParticles * 2)) 
        # ... But building a sparse matrix with 'row-based linked list' is too slow, hence the use of dense matrix for now
        denseA = np.zeros((totalParticles * 2, totalParticles * 2))
        self.b = np.zeros((totalParticles * 2, 1))
        
        ## Assemble A = (M - h * df/dv - h^2 * df/dx)
        # set mass matrix
        for dynamic in scene.dynamics:
            for i in range(dynamic.numParticles):
                massMatrix = np.matlib.identity(2) * dynamic.m[i]
                ids = dynamic.globalOffset + i
                denseA[ids*2:ids*2+2,ids*2:ids*2+2] = massMatrix
        
        # set h^2 * df/dx
        constraintsIterator = scene.getConstraintsIterator()
        for constraint in constraintsIterator:
            ids = constraint.globalIds
            for fi in range(len(ids)):
                for xj in range(len(ids)):
                    Jx = constraint.getJacobianDx(fi, xj)
                    denseA[ids[fi]*2:ids[fi]*2+2,ids[xj]*2:ids[xj]*2+2] -= (Jx * dt * dt)
        
        # set h * df/dv
        constraintsIterator = scene.getConstraintsIterator()
        for constraint in constraintsIterator:
            ids = constraint.globalIds
            for fi in range(len(ids)):
                for vj in range(len(ids)):
                    Jv = constraint.getJacobianDv(fi, vj)
                    denseA[ids[fi]*2:ids[fi]*2+2,ids[vj]*2:ids[vj]*2+2] -= (Jv * dt)
        
        ## Assemble b = h *( f0 + h * df/dx * v0)
        # set (f0 * h)
        for dynamic in scene.dynamics:
            for i in range(dynamic.numParticles):
                ids = dynamic.globalOffset + i
                self.b[ids*2:ids*2+2] += (np.reshape(dynamic.f[i], (2,1)) * dt)

        # set (df/dx * v0 * h * h)
        constraintsIterator = scene.getConstraintsIterator()
        for constraint in constraintsIterator:
            ids = constraint.globalIds
            localIds = constraint.localIds
            for fi in range(len(ids)):
                for xi in range(len(ids)):
                    Jx = constraint.getJacobianDx(fi, xi)
                    self.b[ids[fi]*2:ids[fi]*2+2] += np.reshape(np.matmul(dynamic.v[localIds[xi]], Jx), (2,1)) * dt * dt

        # Convert matrix A to csr matrix (Compressed Sparse Row format) for efficiency reasons
        self.A = sparse.csr_matrix(denseA)

    @profiler.timeit
    def solveSystem(self, scene, dt):
        # Solve the system (Ax=b)
        cgResult = sparse.linalg.cg(self.A, self.b)
        deltaVArray = cgResult[0]
        # Advect
        for dynamic in scene.dynamics:
            for i in range(dynamic.numParticles):
                ids = dynamic.globalOffset + i
                deltaV = [float(deltaVArray[ids*2]), float(deltaVArray[ids*2+1])]
                deltaX = (dynamic.v[i] + deltaV) * dt
                dynamic.v[i] += deltaV
                dynamic.x[i] += deltaX

'''
 Semi Implicit Step
'''
class SemiImplicitSolver(BaseSolver):
    def __init__(self, dt, stepsPerFrame):
        BaseSolver.__init__(self, dt, stepsPerFrame)
    
    @profiler.timeit    
    def prepareSystem(self, scene, dt):
        # Set gravity
        for dynamic in scene.dynamics:
            dynamic.f.fill(0.0)
            for i in range(dynamic.numParticles):
                dynamic.f[i] += np.multiply(scene.gravity, dynamic.m[i])
                
        # Get iterator on constraint to access from objects and scene at once
        constraintsIterator = scene.getConstraintsIterator()
    
        # Compute and add internal forces
        for constraint in constraintsIterator:
            constraint.computeForces(scene)
            constraint.applyForces(scene)

    @profiler.timeit
    def assembleSystem(self, scene, dt):
        pass

    @profiler.timeit
    def solveSystem(self, scene, dt):
        # Integrator
        for dynamic in scene.dynamics:
            for i in range(dynamic.numParticles):
                dynamic.v[i] += dynamic.f[i] * dynamic.im[i] * dt
                dynamic.x[i] += dynamic.v[i] * dt
