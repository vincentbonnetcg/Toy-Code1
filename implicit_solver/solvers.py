"""
@author: Vincent Bonnet
@description : symplectic and backward Euler integrators
"""

import numpy as np
import profiler as profiler
import scipy.sparse as sparse

'''
 Base Solver
'''
class BaseSolver:
    def __init__(self, dt, stepsPerFrame):
        self.dt = dt # delta time per step
        self.stepsPerFrame = stepsPerFrame # number of step per frame

    @profiler.timeit
    def solveFrame(self, scene):
        for substepId in range(self.stepsPerFrame):
            self.step(scene, self.dt)
            
    def step(self, scene, dt):
        self.assembleSystem(scene, dt)
        self.solveSystem(scene, dt)

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
    def assembleSystem(self, scene, dt):
        # Set gravity
        for data in scene.objects:
            data.f.fill(0.0)
            for i in range(data.numParticles):
                data.f[i] += np.multiply(scene.gravity, data.m[i])

        # Prepare forces and jacobians
        for data in scene.objects:
            for constraint in data.constraints:
                constraint.computeForces(scene)
                constraint.computeJacobians(scene)
                constraint.applyForces(scene)
        
        # Assemble the system (Ax=b) where x is the change of velocity
        totalParticles = scene.numParticles()
        # attempt to create 'row-based linked list' sparse matrix for simple sparse matrix construction
        # with A = sparse.lil_matrix((totalParticles * 2, totalParticles * 2)) 
        # ... But building a sparse matrix with 'row-based linked list' is too slow, hence the use of dense matrix for now
        denseA = np.zeros((totalParticles * 2, totalParticles * 2))
        self.b = np.zeros((totalParticles * 2, 1))
        
        ## Assemble A = (M - h * df/dv - h^2 * df/dx)
        # set mass matrix
        for data in scene.objects:
            for i in range(data.numParticles):
                massMatrix = np.matlib.identity(2) * data.m[i]
                ids = data.globalOffset + i
                denseA[ids*2:ids*2+2,ids*2:ids*2+2] = massMatrix
        
        # set h * df/dv
        for data in scene.objects:
            for constraint in data.constraints:
                ids = constraint.globalIds
                for fi in range(len(ids)):
                    for xj in range(len(ids)):
                        Jx = constraint.getJacobianDx(fi, xj)
                        denseA[ids[fi]*2:ids[fi]*2+2,ids[xj]*2:ids[xj]*2+2] -= (Jx * dt * dt)
        
        # set h^2 * df/dx
        for data in scene.objects:
            for constraint in data.constraints:
                ids = constraint.globalIds
                for fi in range(len(ids)):
                    for vj in range(len(ids)):
                        Jv = constraint.getJacobianDv(fi, vj)
                        denseA[ids[fi]*2:ids[fi]*2+2,ids[vj]*2:ids[vj]*2+2] -= (Jv * dt)
        
        ## Assemble b = h *( f0 + h * df/dx * v0)
        # set (f0 * h)
        for data in scene.objects:
            for i in range(data.numParticles):
                ids = data.globalOffset + i
                self.b[ids*2:ids*2+2] += (np.reshape(data.f[i], (2,1)) * dt)

        # set (df/dx * v0 * h * h)
        for constraint in data.constraints:
            ids = constraint.globalIds
            localIds = constraint.ids
            for fi in range(len(ids)):
                for xi in range(len(ids)):
                    Jx = constraint.getJacobianDx(fi, xi)
                    self.b[ids[fi]*2:ids[fi]*2+2] += np.reshape(np.matmul(data.v[localIds[xi]], Jx), (2,1)) * dt * dt

        # Convert matrix A to csr matrix (Compressed Sparse Row format) for efficiency reasons
        self.A = sparse.csr_matrix(denseA)

    @profiler.timeit
    def solveSystem(self, scene, dt):
        # Solve the system (Ax=b)
        cgResult = sparse.linalg.cg(self.A, self.b)
        deltaVArray = cgResult[0]
        # Advect
        for data in scene.objects:
            for i in range(data.numParticles):
                ids = data.globalOffset + i
                deltaV = [float(deltaVArray[ids*2]), float(deltaVArray[ids*2+1])]
                deltaX = (data.v[i] + deltaV) * dt
                data.v[i] += deltaV
                data.x[i] += deltaX

'''
 Semi Implicit Step
'''
class SemiImplicitSolver(BaseSolver):
    def __init__(self, dt, stepsPerFrame):
        BaseSolver.__init__(self, dt, stepsPerFrame)

    @profiler.timeit
    def assembleSystem(self, scene, dt):
        # Set gravity
        for data in scene.objects:
            data.f.fill(0.0)
            for i in range(data.numParticles):
                data.f[i] += np.multiply(scene.gravity, data.m[i])
    
        # Compute and add internal forces
        for data in scene.objects:
            for constraint in data.constraints:
                constraint.computeForces(scene)
                constraint.applyForces(scene)

    @profiler.timeit
    def solveSystem(self, scene, dt):
        # Integrator
        for data in scene.objects:
            for i in range(data.numParticles):
                data.v[i] += data.f[i] * data.im[i] * dt
                data.x[i] += data.v[i] * dt
