"""
@author: Vincent Bonnet
@description : Helper functions for time integrators
"""
import numba # required by core.code_gen
import numpy as np

import core.jit.data_accessor as db
import core.code_gen as generate
import lib.system.jit.sparse_matrix_lib as sparse_lib
from lib.objects.jit.data import Constraint, Node

def apply_external_forces_to_nodes(dynamics, forces):
    # this function is not vectorized but forces.apply_forces are vectorized
    for force in forces:
        force.apply_forces(dynamics)

@generate.vectorize
def reset_forces(node : Node):
    node.f[:] = 0.0

@generate.vectorize
def set_system_index(node : Node, system_index_counter):
    # system_index_counter = np.zeros(1)
    node.systemIndex = system_index_counter[0]
    system_index_counter[0] += 1

@generate.vectorize
def update_system_indices(constraint : Constraint, detail_nodes):
    num_nodes = len(constraint.node_IDs)
    for i in range(num_nodes):
        si = db.systemIndex(detail_nodes, constraint.node_IDs[i])
        constraint.systemIndices[i] = si

@generate.vectorize
def apply_constraint_forces_to_nodes(constraint : Constraint, detail_nodes):
    # Cannot be threaded yet to prevent different threads to write on the same node
    num_nodes = len(constraint.node_IDs)
    for i in range(num_nodes):
        db.add_f(detail_nodes,  constraint.node_IDs[i], constraint.f[i])

@generate.vectorize
def advect(node : Node, delta_v, dt):
    # Can be threaded
    node.v += delta_v[node.systemIndex]
    node.x += node.v * dt

@generate.vectorize
def assemble_fo_h_to_b(node : Node, dt, b):
    # Can be threaded
    b[node.systemIndex] += node.f * dt

@generate.vectorize
def assemble_dfdx_v0_h2_to_b(constraint : Constraint, detail_nodes, dt, b):
    # Cannot be threaded yet
    num_nodes = len(constraint.node_IDs)
    for fi in range(num_nodes):
        node_index = constraint.systemIndices[fi]
        for xi in range(num_nodes):
            Jx = constraint.dfdx[fi][xi]
            v = db.v(detail_nodes, constraint.node_IDs[xi])
            b[node_index] += np.dot(v, Jx) * dt * dt

@generate.vectorize
def assemble_mass_matrix_to_A(node : Node, A):
    # Can be threaded
    system_index = node.systemIndex
    mass_matrix = np.zeros((2,2))
    np.fill_diagonal(mass_matrix, node.m)
    sparse_lib.add(A, system_index, system_index, mass_matrix)

@generate.vectorize
def assemble_constraint_forces_to_A(constraint : Constraint, dt, A):
    # Substract (h * df/dv + h^2 * df/dx)
    # Cannot be threaded yet
    num_nodes = len(constraint.node_IDs)
    for fi in range(num_nodes):
        for j in range(num_nodes):
            Jv = constraint.dfdv[fi][j]
            Jx = constraint.dfdx[fi][j]
            global_fi_id = constraint.systemIndices[fi]
            global_j_id = constraint.systemIndices[j]
            sparse_lib.add(A, global_fi_id, global_j_id, ((Jv * dt) + (Jx * dt * dt)) * -1.0)

@numba.njit
def assemble_A(details,
               num_rows,
               dt,
               mass_matrix_assembly_func,
               constraint_matrix_assembly_func):

    sub_size=2 # submatrix size

    # create mass matrix
    A = sparse_lib.create_empty_sparse_matrix(num_rows, sub_size)
    mass_matrix_assembly_func(details.node, A)

    # assemble constraint in matrix
    constraint_matrix_assembly_func(details.area, dt, A)
    constraint_matrix_assembly_func(details.bending, dt, A)
    constraint_matrix_assembly_func(details.spring, dt, A)
    constraint_matrix_assembly_func(details.anchorSpring, dt, A)

    # allocate and set number of entries per row
    row_indptr = np.empty(num_rows+1, dtype=np.int32)
    row_indptr[0] = 0
    total_entries = 0
    for row_id in range(num_rows):
        num_entries = len(A[row_id])
        total_entries += num_entries
        row_indptr[row_id+1] = total_entries

    # allocate column indices and array of matrix
    data = np.empty((total_entries, sub_size, sub_size))
    column_indices = np.empty(total_entries, dtype=np.int32)

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

    return data, column_indices, row_indptr

@generate.vectorize
def euler_integration(node : Node, dt):
    # Can be threaded
    node.v += node.f * node.im * dt
    node.x += node.v * dt
