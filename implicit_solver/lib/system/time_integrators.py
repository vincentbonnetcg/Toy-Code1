"""
@author: Vincent Bonnet
@description : symplectic and backward Euler integrators
"""

import numpy as np
import scipy
import scipy.sparse
import scipy.sparse.linalg

import lib.common as cm
import lib.common.node_accessor as na
import lib.common.code_gen as generate
import lib.objects.components as cpn

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


'''
Vectorized functions
'''
def apply_external_forces_to_nodes(dynamics, forces):
    # this function is not vectorized but forces.apply_forces are vectorized
    for force in forces:
        force.apply_forces(dynamics)

@generate.as_vectorized
def apply_constraint_forces_to_nodes(constraint : cpn.ConstraintBase, detail_nodes):
    # Cannot be threaded yet to prevent different threads to write on the same node
    num_nodes = len(constraint.node_IDs)
    for i in range(num_nodes):
        na.node_add_f(detail_nodes,  constraint.node_IDs[i], constraint.f[i])

@generate.as_vectorized
def advect(node : cpn.Node, delta_v, dt):
    # Can be threaded
    node_index = na.node_global_index(node.ID)
    node.v += delta_v[node_index]
    node.x += node.v * dt

@generate.as_vectorized
def assemble_fo_h_to_b(node : cpn.Node, dt, b):
    # Can be threaded
    node_index = na.node_global_index(node.ID)
    b[node_index] += node.f * dt

@generate.as_vectorized
def assemble_dfdx_v0_h2_to_b(constraint : cpn.ConstraintBase, detail_nodes, dt, b):
    # Cannot be threaded yet
    num_nodes = len(constraint.node_IDs)
    for fi in range(num_nodes):
        node_index = na.node_global_index(constraint.node_IDs[fi])
        for xi in range(num_nodes):
            Jx = constraint.dfdx[fi][xi]
            v = na.node_v(detail_nodes, constraint.node_IDs[xi])
            b[node_index] += np.dot(v, Jx) * dt * dt

@generate.as_vectorized(njit=False)
def assemble_mass_matrix_to_A(node : cpn.Node, A):
    # Can be threaded
    node_index = na.node_global_index(node.ID)
    mass_matrix = np.zeros((2,2))
    np.fill_diagonal(mass_matrix, node.m)
    A.add(node_index, node_index, mass_matrix)

@generate.as_vectorized(njit=False)
def assemble_constraint_forces_to_A(constraint : cpn.ConstraintBase, dt, A):
    # Substract (h * df/dv + h^2 * df/dx)
    # Cannot be threaded yet
    num_nodes = len(constraint.node_IDs)
    for fi in range(num_nodes):
        for j in range(num_nodes):
            Jv = constraint.dfdv[fi][j]
            Jx = constraint.dfdx[fi][j]
            global_fi_id = na.node_global_index(constraint.node_IDs[fi])
            global_j_id = na.node_global_index(constraint.node_IDs[j])
            A.add(global_fi_id, global_j_id, ((Jv * dt) + (Jx * dt * dt)) * -1.0)


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
            condition.compute_forces(scene, details)
            condition.compute_jacobians(scene, details)

        # Add forces to dynamics
        apply_external_forces_to_nodes(details.dynamics(), scene.forces)
        apply_constraint_forces_to_nodes(details.conditions(), details.node)

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

        # set mass matrix
        assemble_mass_matrix_to_A(details.dynamics(), A)

        # add constraint force to sparse matrix
        assemble_constraint_forces_to_A(details.conditions(), dt, A)

        # convert sparse matrix
        self.A = A.sparse_matrix()

    @cm.timeit
    def _assemble_b(self, details, dt):
        '''
        Assemble b = h *( f0 + h * df/dx * v0)
                 b = (f0 * h) + (h^2 * df/dx * v0)
        '''
        # create b vector
        self.b = np.zeros((self.num_nodes, 2))

        # set (f0 * h)
        assemble_fo_h_to_b(details.dynamics(), dt, self.b)

        # add (df/dx * v0 * h * h)
        assemble_dfdx_v0_h2_to_b(details.conditions(), details.node, dt, self.b)

    @cm.timeit
    def _advect(self, details, delta_v, dt):
        advect(details.dynamics(), delta_v, dt)

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
            condition.compute_forces(scene)

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
