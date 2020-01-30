"""
@author: Vincent Bonnet
@description : Helper functions for time integrators
"""
import numba
import numpy as np

import lib.common.jit.node_accessor as na
import lib.common.code_gen as generate
import lib.objects.components_jit as cpn
from . import sparse_matrix_lib as sparse_lib

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
    sparse_lib.add(A.dict_indices, node_index, node_index, mass_matrix)

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
            sparse_lib.add(A.dict_indices, global_fi_id, global_j_id, ((Jv * dt) + (Jx * dt * dt)) * -1.0)
