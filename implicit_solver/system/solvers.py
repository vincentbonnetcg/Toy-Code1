"""
@author: Vincent Bonnet
@description : symplectic and backward Euler integrators
"""

import numpy as np
import scipy as sc
import scipy.sparse
import scipy.sparse.linalg
from common import profiler
from system.sparse_matrix import BSRSparseMatrix, DebugSparseMatrix

'''
 Base Solver
'''
class BaseSolver:
    def __init__(self):
        self.currentTime = 0.0

    def initialize(self, scene):
        '''
        Initialize the solver and the data used by the solver
        '''
        self.currentTime = 0.0
        scene.updateKinematics(self.currentTime)
        scene.updateConditions(True) # Update static conditions

    @profiler.timeit
    def solveStep(self, scene, dt):
        self.currentTime += dt
        self.preStep(scene, self.currentTime)
        self.step(scene, dt)

    def preStep(self, scene, time):
        scene.updateKinematics(time)
        scene.updateConditions(False) # Update dynamic conditions

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
    def __init__(self):
        BaseSolver.__init__(self)
        # used to store system Ax=b
        self.A = None
        self.b = None

    @profiler.timeit
    def prepareSystem(self, scene, dt):
        # Set gravity
        for dynamic in scene.dynamics:
            dynamic.f.fill(0.0)
            for i in range(dynamic.num_particles):
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
        num_rows = totalParticles
        num_columns = totalParticles
        A = BSRSparseMatrix(num_rows, num_columns, 2)

        ## Assemble A = (M - h * df/dv - h^2 * df/dx)
        ## => Assemble A = (M - (h * df/dv + h^2 * df/dx))
        # set mass matrix
        for dynamic in scene.dynamics:
            for i in range(dynamic.num_particles):
                mass_matrix = np.zeros((2,2))
                np.fill_diagonal(mass_matrix, dynamic.m[i])
                idx = dynamic.global_offset + i

                A.add(idx, idx, mass_matrix)

        # Substract (h * df/dv + h^2 * df/dx)
        constraintsIterator = scene.getConstraintsIterator()
        for constraint in constraintsIterator:
            ids = constraint.globalIds
            for fi in range(len(ids)):
                for j in range(len(ids)):
                    Jv = constraint.getJacobianDv(fi, j)
                    Jx = constraint.getJacobianDx(fi, j)

                    A.add(ids[fi], ids[j], ((Jv * dt) + (Jx * dt * dt)) * -1.0)

        ## Assemble b = h *( f0 + h * df/dx * v0)
        # set (f0 * h)
        self.b = np.zeros(num_columns * 2)
        for dynamic in scene.dynamics:
            for i in range(dynamic.num_particles):
                idx = dynamic.global_offset + i
                self.b[idx*2:idx*2+2] += dynamic.f[i] * dt

        # set (df/dx * v0 * h * h)
        constraintsIterator = scene.getConstraintsIterator()
        for constraint in constraintsIterator:
            ids = constraint.globalIds
            localIds = constraint.localIds
            dynamicIndices = constraint.dynamicIndices
            for fi in range(len(ids)):
                for xi in range(len(ids)):
                    dynamic = scene.dynamics[dynamicIndices[xi]]
                    Jx = constraint.getJacobianDx(fi, xi)
                    self.b[ids[fi]*2:ids[fi]*2+2] += np.matmul(dynamic.v[localIds[xi]], Jx) * dt * dt

        # convert sparse matrix
        self.A = A.sparse_matrix()

    @profiler.timeit
    def solveSystem(self, scene, dt):
        # Solve the system (Ax=b)
        cgResult = sc.sparse.linalg.cg(self.A, self.b)
        deltaVArray = cgResult[0]
        # Advect
        for dynamic in scene.dynamics:
            for i in range(dynamic.num_particles):
                ids = dynamic.global_offset + i
                deltaV = [float(deltaVArray[ids*2]), float(deltaVArray[ids*2+1])]
                deltaX = (dynamic.v[i] + deltaV) * dt
                dynamic.v[i] += deltaV
                dynamic.x[i] += deltaX

'''
 Semi Implicit Step
'''
class SemiImplicitSolver(BaseSolver):
    def __init__(self):
        BaseSolver.__init__(self)

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
