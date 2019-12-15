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
@generate.as_vectorized
def advect(node : cpn.Node, delta_vs, dt):
    node_index = na.node_global_index(node.ID)
    delta_v = delta_vs[node_index]
    node.x += (node.v + delta_v) * dt
    node.v += delta_v

@generate.as_vectorized
def assemble_b__fo_h(node : cpn.Node, b, dt):
    offset = na.node_global_index(node.ID) * 2
    b[offset:offset+2] += node.f * dt


#@generate.as_vectorized
#def dfdx_v0_h2(cnt : cpn.ConstraintBase, details, b, dt):
#    num_nodes = len(cnt.node_ids)
#    for fi in range(num_nodes):
#        for xi in range(num_nodes):
#            Jx = cnt.dfdx[fi][xi]
#            x, v = na.node_xv(details.node, cnt.node_ids[xi])
#            b_offset = na.node_global_index(cnt.node_ids[fi]) * 2
#            b[b_offset:b_offset+2] += np.dot(v, Jx) * dt * dt

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

    @cm.timeit
    def prepare_system(self, scene, details, dt):
        # Reset forces on all dynamic
        for dynamic in scene.dynamics:
            details.node.fill('f', 0.0, dynamic.block_ids)

        # Prepare external forces
        for force in scene.forces:
            force.apply_forces(details)

        # Prepare constraints (forces and jacobians)
        for condition in scene.conditions:
            condition.compute_forces(scene, details)
            condition.compute_jacobians(scene, details)

        # Add forces to object from constraints
        for condition in scene.conditions:
            condition.apply_forces(details)

    @cm.timeit
    def assemble_system(self, details, dt):
        '''
        Assemble the system (Ax=b) where x is the unknow change of velocity
        '''
        if (details.node.compute_num_elements() == 0):
            return

        self._assemble_A(details, dt)
        self._assemble_b(details, dt)

    @cm.timeit
    def solve_system(self, details, dt):
        num_nodes = details.node.compute_num_elements()
        if (num_nodes == 0):
            return

        # Solve the system (Ax=b) and reshape the conjugate gradient result
        # In this case, the reshape operation is not causing any reallocation
        cg_result = scipy.sparse.linalg.cg(self.A, self.b)
        delta_v = cg_result[0].reshape(num_nodes, 2)
        # Advect
        self._advect(details, delta_v, dt)

    @cm.timeit
    def _assemble_A(self, details, dt):
        '''
        Assemble A = (M - (h * df/dv + h^2 * df/dx))
        '''
        total_nodes = details.node.compute_num_elements()
        num_rows = total_nodes
        num_columns = total_nodes
        A = cm.BSRSparseMatrix(num_rows, num_columns, 2)

        # Set mass matrix
        for dynamic in details.dynamics():
            for obj_block in dynamic.blocks:
                data_m = obj_block['m']
                data_node_id = obj_block['ID']
                block_n_elements = obj_block['blockInfo_numElements']
                for i in range(block_n_elements):
                    mass_matrix = np.zeros((2,2))
                    np.fill_diagonal(mass_matrix, data_m[i])
                    idx = na.node_global_index(data_node_id[i])
                    A.add(idx, idx, mass_matrix)

        # Substract (h * df/dv + h^2 * df/dx)
        for condition in details.conditions():
            for ct_block in condition.blocks:
                node_ids_ptr = ct_block['node_IDs']
                dfdv_ptr = ct_block['dfdv']
                dfdx_ptr = ct_block['dfdx']
                block_n_elements = ct_block['blockInfo_numElements']
                for cid in range(block_n_elements):
                    ids = node_ids_ptr[cid]
                    for fi in range(len(ids)):
                        for j in range(len(ids)):
                            Jv = dfdv_ptr[cid][fi][j]
                            Jx = dfdx_ptr[cid][fi][j]
                            global_fi_id = na.node_global_index(ids[fi])
                            global_j_id = na.node_global_index(ids[j])
                            A.add(global_fi_id, global_j_id, ((Jv * dt) + (Jx * dt * dt)) * -1.0)

        # convert sparse matrix
        self.A = A.sparse_matrix()

    @cm.timeit
    def _assemble_b(self, details, dt):
        '''
        Assemble b = h *( f0 + h * df/dx * v0)
                 b = (f0 * h) + (h^2 * df/dx * v0)
        '''
        total_nodes = details.node.compute_num_elements()
        self.b = np.zeros(total_nodes * 2)

        # set (f0 * h)
        assemble_b__fo_h(details.node, self.b, dt)

        # add (df/dx * v0 * h * h)
        #dfdx_v0_h2(details.conditions, details.dynamics, self.b, dt)

        # add (df/dx * v0 * h * h)
        for condition in details.conditions():
            for ct_block in condition.blocks:
                node_ids_ptr = ct_block['node_IDs']
                dfdx_ptr = ct_block['dfdx']
                block_n_elements = ct_block['blockInfo_numElements']
                for cid in range(block_n_elements):
                    ids = node_ids_ptr[cid]
                    for fi in range(len(ids)):
                        for xi in range(len(ids)):
                            Jx = dfdx_ptr[cid][fi][xi]
                            v = na.node_v(details.node, ids[xi])
                            vec = np.dot(v, Jx) * dt * dt
                            b_offset = na.node_global_index(ids[fi]) * 2
                            self.b[b_offset:b_offset+2] += vec

    @cm.timeit
    def _advect(self, details, delta_v, dt):
        advect(details.node, delta_v, dt)

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
            condition.apply_forces(scene.dynamics)

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
