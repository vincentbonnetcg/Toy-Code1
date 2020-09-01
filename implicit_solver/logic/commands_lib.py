"""
@author: Vincent Bonnet
@description : commands to setup objects and run simulation
"""

import lib.objects as objects
import lib.common as cm
import lib.common.jit.data_accessor as db
import numpy as np

def set_render_prefs(obj, prefs):
    # Render preferences used by render.py
    obj.meta_data['render_prefs'] = prefs

def add_kinematic(scene, details, shape, animator = None):
    kinematic = objects.Kinematic(details, shape)
    scene.add_kinematic(kinematic, animator)
    return kinematic

def add_dynamic(scene, details, shape, node_mass):
    dynamic = objects.Dynamic(details, shape, node_mass)
    scene.add_dynamic(dynamic)
    return dynamic

def initialize(scene, solver, details, context):
    solver.initialize(scene, details, context)

@cm.timeit
def solve_to_next_frame(scene, solver, details, context):
    for _ in range(context.num_substep):
        context.time += context.dt
        solver.solve_step(scene, details, context)

def get_nodes_from_dynamic(dynamic, details):
    return details.db['node'].flatten('x', dynamic.block_handles)

def get_shape_from_kinematic(kinematic, details):
    return kinematic.get_as_shape(details)

def get_normals_from_kinematic(kinematic, details, normal_scale=0.2):
    segs = []
    normals = details.db['edge'].flatten('normal', kinematic.edge_handles)
    point_IDs = details.db['edge'].flatten('point_IDs', kinematic.edge_handles)
    num_normals = len(normals)

    for i in range(num_normals):
        x0 = db.x(details.point, point_IDs[i][0])
        x1 = db.x(details.point, point_IDs[i][1])
        points = [None, None]
        points[0] = (x0+x1)*0.5
        points[1] = points[0]+(normals[i]*normal_scale)
        segs.append(points)

    return segs

def get_segments_from_constraint(condition, details):
    segs = []

    condition_data = details.datablock_from_typename(condition.typename)

    node_ids = condition_data.flatten('node_IDs', condition.block_handles)
    num_constraints = len(node_ids)
    for ct_index in range(num_constraints):
        num_nodes = len(node_ids[ct_index])
        if num_nodes == 2:
            points = []
            for node_index in range (num_nodes):
                x = db.x(details.node, node_ids[ct_index][node_index])
                points.append(x)
            segs.append(points)

    return segs

def get_sparse_matrix_as_dense(details, solver, as_binary=False):
    if hasattr(solver.time_integrator, 'A'):
        A = solver.time_integrator.A
        if A is None:
            return None

        denseA = np.abs(A.toarray())
        if as_binary:
            denseA[:] = denseA[:]>0.0
            return denseA

        return denseA

    return None
