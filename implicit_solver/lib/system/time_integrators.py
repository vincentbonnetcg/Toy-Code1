"""
@author: Vincent Bonnet
@description : symplectic and backward Euler integrators
"""

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

import lib.common as cm
import lib.system.jit.integrator_lib as integrator_lib
import lib.system.jit.sparse_matrix_lib as sparse_lib

class TimeIntegrator:
    '''
    Base class for time integrator
    '''
    def prepare_system(self, scene, details, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'prepare_system'")

    def assemble_system(self, details, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'assemble_system'")

    def solve_system(self, details, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'solve_system'")


class ImplicitSolver(TimeIntegrator):
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
        details.node.fill('f', 0.0)

        # Compute constraint forces and jacobians
        for condition in scene.conditions:
            condition.pre_compute(scene, details)
            condition.compute_gradients(details)
            condition.compute_hessians(details)

        # Add forces to dynamics
        integrator_lib.apply_external_forces_to_nodes(details.dynamics(), scene.forces)
        integrator_lib.apply_constraint_forces_to_nodes(details.conditions(), details.node)

        # Store number of nodes
        self.num_nodes = details.node.compute_num_elements()

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
        num_columns = self.num_nodes
        A = cm.BSRSparseMatrix(num_rows, num_columns, 2)

        # TODO : SuperUgly but issue when using Numba 0.48.0 to lower details class
        node_blocks = details.node.blocks
        area_blocks = details.area.blocks
        bending_blocks = details.bending.blocks
        spring_blocks = details.spring.blocks
        anchorSpring_blocks = details.anchorSpring.blocks

        num_entries_per_row, A.dict_indices = integrator_lib.assemble_A(node_blocks,
                                                   area_blocks,
                                                   bending_blocks,
                                                   spring_blocks,
                                                   anchorSpring_blocks,
                                                   num_rows,
                                                   dt,
                                                   integrator_lib.assemble_mass_matrix_to_A.function,
                                                   integrator_lib.assemble_constraint_forces_to_A.function)

        # convert sparse matrix
        self.A = A.sparse_matrix(num_entries_per_row)

    @cm.timeit
    def _assemble_b(self, details, dt):
        '''
        Assemble b = h *( f0 + h * df/dx * v0)
                 b = (f0 * h) + (h^2 * df/dx * v0)
        '''
        # create b vector
        self.b = np.zeros((self.num_nodes, 2))

        # set (f0 * h)
        integrator_lib.assemble_fo_h_to_b(details.dynamics(), dt, self.b)

        # add (df/dx * v0 * h * h)
        integrator_lib.assemble_dfdx_v0_h2_to_b(details.conditions(), details.node, dt, self.b)

    @cm.timeit
    def _advect(self, details, delta_v, dt):
        integrator_lib.advect(details.dynamics(), delta_v, dt)

'''
NEED TO RE-IMPLEMENT
class SemiImplicitSolver(TimeIntegrator):
    def __init__(self):
        Solver.__init__(self)

    @cm.timeit
    def prepare_system(self, scene, details, dt):
        # Reset forces
        for dynamic in scene.dynamics:
            dynamic.data.fill('f', 0.0)

        # Apply external forces
        for force in scene.forces:
            force.apply_forces(scene.dynamics)

        # Apply internal forces
        for condition in scene.conditions:
            condition.compute_gradients(scene)

        apply_constraint_forces_to_nodes(details.conditions(), details.node)

    @cm.timeit
    def assemble_system(self, details, dt):
        pass

    @cm.timeit
    def solve_system(self, details, dt):
        # Integrator
        for dynamic in scene.dynamics:
            for i in range(dynamic.num_nodes()):
                dynamic.v[i] += dynamic.f[i] * dynamic.im[i] * dt
                dynamic.x[i] += dynamic.v[i] * dt
'''
