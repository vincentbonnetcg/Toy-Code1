"""
@author: Vincent Bonnet
@description : symplectic and backward Euler integrators
"""

import numpy as np
import scipy as sc
import scipy.sparse
import scipy.sparse.linalg
from tools import profiler
from system.sparse_matrix import BSRSparseMatrix, DebugSparseMatrix

class Context:
    '''
    Context to store time, time stepping, etc.
    '''
    def __init__(self, time = 0.0, frame_dt = 1.0/24.0, num_substep = 4, num_frames = 100):
        self.time = time # current time (in seconds)
        self.start_time = time # start time (in seconds)
        self.end_time = time + (num_frames * frame_dt) # end time (in seconds)
        self.frame_dt = frame_dt # time step on a single frame (in seconds)
        self.num_substep = num_substep # number of substep per frame
        self.dt = frame_dt / num_substep # simulation substep (in seconds)
        self.num_frames = num_frames # number of simulated frame (doesn't include initial frame)

class BaseSolver:
    '''
    Base Solver
    '''
    def __init__(self):
        pass

    def initialize(self, scene, context):
        '''
        Initialize the solver and the data used by the solver
        '''
        scene.update_kinematics(context.time)
        scene.update_conditions(True) # Update static conditions
        scene.update_conditions(False) # Update dynamic conditions

    @profiler.timeit
    def solveStep(self, scene, context):
        self.preStep(scene, context)
        self.step(scene, context)
        self.postStep(scene, context)

    def preStep(self, scene, context):
        scene.update_kinematics(context.time, context.dt)
        scene.update_conditions(False) # Update dynamic conditions

    def step(self, scene, context):
        self.prepareSystem(scene, context.dt)
        self.assembleSystem(scene, context.dt)
        self.solveSystem(scene, context.dt)

    def postStep(self, scene, context):
        pass

    def prepareSystem(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'prepareSystem'")

    def assembleSystem(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'assembleSystem'")

    def solveSystem(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'solveSystem'")


class ImplicitSolver(BaseSolver):
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
    def __init__(self):
        BaseSolver.__init__(self)
        # used to store system Ax=b
        self.A = None
        self.b = None

    @profiler.timeit
    def prepareSystem(self, scene, dt):
        # Reset forces
        for dynamic in scene.dynamics:
            dynamic.f.fill(0.0)

        # Prepare external forces
        for force in scene.forces:
            force.apply_forces(scene)

        # Prepare constraints (forces and jacobians)
        constraintsIterator = scene.get_constraints_iterator()
        for constraint in constraintsIterator:
            constraint.computeForces(scene)
            constraint.computeJacobians(scene)
            constraint.applyForces(scene)

    @profiler.timeit
    def assembleSystem(self, scene, dt):
        # Assemble the system (Ax=b) where x is the change of velocity
        total_particles = scene.num_particles()
        num_rows = total_particles
        num_columns = total_particles
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
        constraintsIterator = scene.get_constraints_iterator()
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
        constraintsIterator = scene.get_constraints_iterator()
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
            v = dynamic.v
            x = dynamic.x
            for i in range(dynamic.num_particles):
                ids = dynamic.global_offset + i
                deltaV = [float(deltaVArray[ids*2]), float(deltaVArray[ids*2+1])]
                deltaX = (v[i] + deltaV) * dt
                v[i] += deltaV
                x[i] += deltaX

class SemiImplicitSolver(BaseSolver):
    '''
     Semi Implicit Step
    '''
    def __init__(self):
        BaseSolver.__init__(self)

    @profiler.timeit
    def prepareSystem(self, scene, dt):
        # Reset forces
        for dynamic in scene.dynamics:
            dynamic.f.fill(0.0)

        # Apply external forces
        for force in scene.forces:
            force.apply_forces(scene)

        # Apply internal forces
        constraintsIterator = scene.get_constraints_iterator()
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
            for i in range(dynamic.num_particles):
                dynamic.v[i] += dynamic.f[i] * dynamic.im[i] * dt
                dynamic.x[i] += dynamic.v[i] * dt
