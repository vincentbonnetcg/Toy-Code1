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
    def prepare_system(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'prepare_system'")

    def assemble_system(self, scene, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'assemble_system'")

    def solve_system(self, scene, dt):
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
#def dfdx_v0_h2(cnt : cpn.ConstraintBase, dynamics, b, dt):
#    num_nodes = len(cnt.node_ids)
#    for fi in range(num_nodes):
#        for xi in range(num_nodes):
#            Jx = cnt.dfdx[fi][xi]
#            x, v = na.node_xv(dynamics, cnt.node_ids[xi])
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
    def prepare_system(self, scene, dt):
        # Reset forces
        for dynamic in scene.dynamics:
            dynamic.data.fill('f', 0.0)

        # Prepare external forces
        for force in scene.forces:
            force.apply_forces(scene.dynamics)

        # Prepare constraints (forces and jacobians)
        for condition in scene.conditions:
            condition.compute_forces(scene)
            condition.compute_jacobians(scene)

        # Add forces to object from constraints
        for condition in scene.conditions:
            condition.apply_forces(scene.dynamics)

    @cm.timeit
    def assemble_system(self, scene, dt):
        '''
        Assemble the system (Ax=b) where x is the unknow change of velocity
        '''
        self._assemble_A(scene, dt)
        self._assemble_b(scene, dt)

    @cm.timeit
    def solve_system(self, scene, dt):
        if (scene.num_nodes() == 0):
            return
        # Solve the system (Ax=b) and reshape the conjugate gradient result
        # In this case, the reshape operation is not causing any reallocation
        cg_result = scipy.sparse.linalg.cg(self.A, self.b)
        delta_v = cg_result[0].reshape(scene.num_nodes(), 2)
        # Advect
        self._advect(scene, delta_v, dt)

    @cm.timeit
    def _assemble_A(self, scene, dt):
        '''
        Assemble A = (M - (h * df/dv + h^2 * df/dx))
        '''
        total_nodes = scene.num_nodes()
        if (total_nodes == 0):
            return

        num_rows = total_nodes
        num_columns = total_nodes
        A = cm.BSRSparseMatrix(num_rows, num_columns, 2)

        # Set mass matrix
        for dynamic in scene.dynamics:
            data_m = dynamic.data.flatten('m')
            data_node_id = dynamic.data.flatten('ID')
            for i in range(dynamic.num_nodes()):
                mass_matrix = np.zeros((2,2))
                np.fill_diagonal(mass_matrix, data_m[i])
                idx = na.node_global_index(data_node_id[i])
                A.add(idx, idx, mass_matrix)

        # Substract (h * df/dv + h^2 * df/dx)
        for condition in scene.conditions:
            data_node_ids = condition.data.flatten('node_IDs')
            data_dfdv = condition.data.flatten('dfdv')
            data_dfdx = condition.data.flatten('dfdx')
            for cid in range(condition.num_constraints()):
                ids = data_node_ids[cid]
                for fi in range(len(ids)):
                    for j in range(len(ids)):
                        Jv = data_dfdv[cid][fi][j]
                        Jx = data_dfdx[cid][fi][j]
                        global_fi_id = na.node_global_index(ids[fi])
                        global_j_id = na.node_global_index(ids[j])
                        A.add(global_fi_id, global_j_id, ((Jv * dt) + (Jx * dt * dt)) * -1.0)

        # convert sparse matrix
        self.A = A.sparse_matrix()

    @cm.timeit
    def _assemble_b(self, scene, dt):
        '''
        Assemble b = h *( f0 + h * df/dx * v0)
                 b = (f0 * h) + (h^2 * df/dx * v0)
        '''
        total_nodes = scene.num_nodes()
        if (total_nodes == 0):
            return

        self.b = np.zeros(total_nodes * 2)

        # set (f0 * h)
        assemble_b__fo_h(scene.dynamics, self.b, dt)

        # add (df/dx * v0 * h * h)
        #dfdx_v0_h2(scene.conditions, scene.dynamics, self.b, dt)

        # add (df/dx * v0 * h * h)
        for condition in scene.conditions:
            data_node_ids = condition.data.flatten('node_IDs')
            data_dfdx = condition.data.flatten('dfdx')
            for cid in range(condition.num_constraints()):
                ids = data_node_ids[cid]
                for fi in range(len(ids)):
                    for xi in range(len(ids)):
                        Jx = data_dfdx[cid][fi][xi]
                        v = na.node_v(scene.dynamics, ids[xi])
                        vec = np.dot(v, Jx) * dt * dt
                        b_offset = na.node_global_index(ids[fi]) * 2
                        self.b[b_offset:b_offset+2] += vec

    @cm.timeit
    def _advect(self, scene, delta_v, dt):
        advect(scene.dynamics, delta_v, dt)

'''
NEED TO RE-IMPLEMENT
class SemiImplicitSolver(TimeIntegrator):
    def __init__(self):
        Solver.__init__(self)

    @cm.timeit
    def prepare_system(self, scene, dt):
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
    def assemble_system(self, scene, dt):
        pass

    @cm.timeit
    def solve_system(self, scene, dt):
        # Integrator
        for dynamic in scene.dynamics:
            for i in range(dynamic.num_nodes()):
                dynamic.v[i] += dynamic.f[i] * dynamic.im[i] * dt
                dynamic.x[i] += dynamic.v[i] * dt
'''
