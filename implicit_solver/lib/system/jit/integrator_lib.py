"""
@author: Vincent Bonnet
@description : Helper functions for time integrators
"""
import numba # required by lib.common.code_gen
import numpy as np

import lib.common.jit.node_accessor as na
import lib.common.code_gen as generate
import lib.objects.jit as cpn
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

@generate.as_vectorized
def assemble_mass_matrix_to_A(node : cpn.Node, A):
    # Can be threaded
    node_index = na.node_global_index(node.ID)
    mass_matrix = np.zeros((2,2))
    np.fill_diagonal(mass_matrix, node.m)
    sparse_lib.add(A, node_index, node_index, mass_matrix)

@generate.as_vectorized
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
            sparse_lib.add(A, global_fi_id, global_j_id, ((Jv * dt) + (Jx * dt * dt)) * -1.0)

@numba.njit
def assemble_A(node_blocks,
               area_blocks,
               bending_blocks,
               spring_blocks,
               anchorSpring_blocks,
               num_rows,
               dt,
               mass_matrix_assembly_func,
               constraint_matrix_assembly_func):

    sub_size=2 # submatrix size

    # create mass matrix
    A = sparse_lib.create_empty_sparse_matrix(num_rows, sub_size)
    mass_matrix_assembly_func(node_blocks, A)

    # assemble constraint in matrix
    constraint_matrix_assembly_func(area_blocks, dt, A)
    constraint_matrix_assembly_func(bending_blocks, dt, A)
    constraint_matrix_assembly_func(spring_blocks, dt, A)
    constraint_matrix_assembly_func(anchorSpring_blocks, dt, A)

    # allocate and set number of entries per row
    num_entries_per_row = np.zeros(num_rows, dtype=np.int32)
    for row_id in range(num_rows):
        num_entries_per_row[row_id] = len(A[row_id])

    # allocate column indices and array of matrix
    total_entries = np.sum(num_entries_per_row)
    data = np.zeros((total_entries, sub_size, sub_size))
    column_indices = np.zeros(total_entries, dtype=np.int32)

    # set column indices and array of matrix
    idx = 0
    for row_id in range(num_rows):
        # numba-0.0.47 use return an array of key when sorting a dictionnary
        # it should be fix in later version
        sortedA = sorted(A[row_id])
        for column_id in sortedA:
            matrix = A[row_id][column_id]
            column_indices[idx] = column_id
            data[idx] = matrix
            idx += 1

    return num_entries_per_row, column_indices, data

