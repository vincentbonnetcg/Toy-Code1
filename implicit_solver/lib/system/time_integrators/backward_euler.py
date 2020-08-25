"""
@author: Vincent Bonnet
@description : Backward Euler time integrator
"""
import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

import lib.common as cm
import lib.common.jit.block_utils as block_utils
import lib.system.jit.integrator_lib as integrator_lib
from lib.system.time_integrators import TimeIntegrator

class BackwardEulerIntegrator(TimeIntegrator):
    '''
     Implicit Step
     Solve :
         (M - h * df/dv - h^2 * df/dx) * deltaV = h * (f0 + h * df/dx * v0)
           A = (M - h^2 * df/dx)
           b = h * (f0 + h * df/dx * v0)
         => A * deltaV = b <=> deltaV = A^-1 * b
         deltaX = (v0 + deltaV) * h
         v = v + deltaV
         x = x + deltaX
    '''
    def __init__(self):
        TimeIntegrator.__init__(self)
        # used to store system Ax=b
        self.A = None
        self.b = None
        self.num_nodes = 0

    @cm.timeit
    def prepare_system(self, scene, details, dt):
        '''
        Compute external and constraint forces
        '''
        # Reset forces on dynamics
        integrator_lib.reset_forces(details.dynamics)

        # Compute constraint forces and jacobians
        for condition in scene.conditions:
            condition.pre_compute(details.bundle)
            condition.compute_gradients(details.bundle)
            condition.compute_hessians(details.bundle)
            condition.compute_forces(details.bundle)
            condition.compute_force_jacobians(details.bundle)

        # Add forces to dynamics
        integrator_lib.apply_external_forces_to_nodes(details.dynamics, scene.forces)
        integrator_lib.apply_constraint_forces_to_nodes(details.constraints, details.node)

        # Set system index
        system_index_counter = np.zeros(1, dtype = np.int32) # use array to pass value as reference
        integrator_lib.set_system_index(details.dynamics, system_index_counter)
        integrator_lib.update_system_indices(details.constraints, details.node)

        # Store number of nodes
        self.num_nodes = block_utils.compute_num_elements(details.node)

    @cm.timeit
    def assemble_system(self, details, dt):
        '''
        Assemble the system (Ax=b) where x is the unknow change of velocity
        '''
        if (self.num_nodes == 0):
            return

        self._assemble_A(details, dt)
        self._assemble_b(details, dt)

    @cm.timeit
    def solve_system(self, details, dt):
        '''
        Solve the assembled linear system (Ax=b)
        '''
        if (self.num_nodes == 0):
            return

        # Solve the system (Ax=b) and reshape the conjugate gradient result
        # In this case, the reshape operation is not causing any reallocation
        b = self.b.reshape(self.num_nodes * 2)
        cg_result = scipy.sparse.linalg.cg(self.A, b)
        delta_v = cg_result[0].reshape(self.num_nodes, 2)
        # Advect
        self._advect(details, delta_v, dt)

    @cm.timeit
    def _assemble_A(self, details, dt):
        '''
        Assemble A = (M - (h * df/dv + h^2 * df/dx))
        '''
        # create empty sparse matrix A
        num_rows = self.num_nodes
        data, column_indices, row_indptr = integrator_lib.assemble_A(details.bundle,
                                                    num_rows, dt,
                                                    integrator_lib.assemble_mass_matrix_to_A.function,
                                                    integrator_lib.assemble_constraint_forces_to_A.function)

        self.A = scipy.sparse.bsr_matrix((data, column_indices, row_indptr))

    @cm.timeit
    def _assemble_b(self, details, dt):
        '''
        Assemble b = h *( f0 + h * df/dx * v0)
                 b = (f0 * h) + (h^2 * df/dx * v0)
        '''
        # create b vector
        self.b = np.zeros((self.num_nodes, 2))

        # set (f0 * h)
        integrator_lib.assemble_fo_h_to_b(details.dynamics, dt, self.b)

        # add (df/dx * v0 * h * h)
        integrator_lib.assemble_dfdx_v0_h2_to_b(details.constraints, details.node, dt, self.b)

    @cm.timeit
    def _advect(self, details, delta_v, dt):
        integrator_lib.advect(details.dynamics, delta_v, dt)

