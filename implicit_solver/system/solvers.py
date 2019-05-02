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
    def __init__(self, time = 0.0, frame_dt = 1.0/24.0, num_substep = 4, num_frames = 1):
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
    def solve_step(self, scene, context):
        self.pre_step(scene, context)
        self.step(scene, context)
        self.post_step(scene, context)

    def pre_step(self, scene, context):
        scene.update_kinematics(context.time, context.dt)
        scene.update_conditions(False) # Update dynamic conditions

    def step(self, scene, context):
        self.prepare_system(scene, context.dt)
        self.assemble_system(scene, context.dt)
        self.solve_system(scene, context.dt)

    def post_step(self, scene, context):
        pass

    def prepare_system(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'prepare_system'")

    def assemble_system(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'assemble_system'")

    def solve_system(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'solve_system'")


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
    def prepare_system(self, scene, dt):
        # Reset forces
        for dynamic in scene.dynamics:
            dynamic.f.fill(0.0)

        # Prepare external forces
        for force in scene.forces:
            force.apply_forces(scene)

        # Prepare constraints (forces and jacobians)
        constraints_iterator = scene.get_constraints_iterator()
        for constraint in constraints_iterator:
            constraint.compute_forces(scene)
            constraint.compute_jacobians(scene)
            constraint.apply_forces(scene)

    @profiler.timeit
    def assemble_system(self, scene, dt):
        total_particles = scene.num_particles()
        if (total_particles == 0):
            return
        # Assemble the system (Ax=b) where x is the change of velocity
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
        constraints_iterator = scene.get_constraints_iterator()
        for constraint in constraints_iterator:
            ids = constraint.global_particle_ids
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
        constraints_iterator = scene.get_constraints_iterator()
        for constraint in constraints_iterator:
            g_pids = constraint.global_particle_ids
            for fi in range(len(g_pids)):
                for xi in range(len(g_pids)):
                    Jx = constraint.getJacobianDx(fi, xi)
                    x, v = scene.n_state(constraint.n_ids[xi])
                    self.b[g_pids[fi]*2:g_pids[fi]*2+2] += np.matmul(v, Jx) * dt * dt

        # convert sparse matrix
        self.A = A.sparse_matrix()

    @profiler.timeit
    def solve_system(self, scene, dt):
        total_particles = scene.num_particles()
        if (total_particles == 0):
            return
        # Solve the system (Ax=b)
        cgResult = sc.sparse.linalg.cg(self.A, self.b)
        delta_v = cgResult[0]
        # Advect
        self.advect(scene, delta_v, dt)

    @profiler.timeit
    def advect(self, scene, delta_v, dt):
        for dynamic in scene.dynamics:
            v = dynamic.v
            x = dynamic.x
            for i in range(dynamic.num_particles):
                ids = dynamic.global_offset + i
                deltaV = [float(delta_v[ids*2]), float(delta_v[ids*2+1])]
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
    def prepare_system(self, scene, dt):
        # Reset forces
        for dynamic in scene.dynamics:
            dynamic.f.fill(0.0)

        # Apply external forces
        for force in scene.forces:
            force.apply_forces(scene)

        # Apply internal forces
        constraints_iterator = scene.get_constraints_iterator()
        for constraint in constraints_iterator:
            constraint.compute_forces(scene)
            constraint.apply_forces(scene)

    @profiler.timeit
    def assemble_system(self, scene, dt):
        pass

    @profiler.timeit
    def solve_system(self, scene, dt):
        # Integrator
        for dynamic in scene.dynamics:
            for i in range(dynamic.num_particles):
                dynamic.v[i] += dynamic.f[i] * dynamic.im[i] * dt
                dynamic.x[i] += dynamic.v[i] * dt
