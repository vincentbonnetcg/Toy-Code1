"""
@author: Vincent Bonnet
@description : commands to setup objects and run simulation
"""

import lib.objects as objects
import lib.common as cm
import lib.common.jit.node_accessor as na

def set_render_prefs(obj, prefs):
    # Render preferences used by render.py
    obj.meta_data['render_prefs'] = prefs

def add_kinematic(scene, details, shape, position = (0., 0.), rotation = 0., animator = None):
    kinematic = objects.Kinematic(details, shape, position, rotation)
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

def get_nodes_from_dynamic(scene, index, details):
    dynamic = scene.dynamics[index]
    return details.node.flatten('x', dynamic.block_handles)

def get_shape_from_kinematic(scene, index, details):
    kinematic = scene.kinematics[index]
    return kinematic.get_as_shape(details)

def get_normals_from_kinematic(scene, index, details, normal_scale=0.2):
    segs = []
    kinematic = scene.kinematics[index]
    normals = details.edge.flatten('normal', kinematic.edge_handles)
    point_IDs = details.edge.flatten('point_IDs', kinematic.edge_handles)
    num_normals = len(normals)

    for i in range(num_normals):
        x0 = na.node_x(details.point.blocks, point_IDs[i][0])
        x1 = na.node_x(details.point.blocks, point_IDs[i][1])
        points = [None, None]
        points[0] = (x0+x1)*0.5
        points[1] = points[0]+(normals[i]*normal_scale)
        segs.append(points)

    return segs

def get_segments_from_constraint(scene, index, details):
    segs = []

    condition = scene.conditions[index]
    condition_data = details.block_from_datatype(condition.constraint_type)

    node_ids = condition_data.flatten('node_IDs', condition.block_handles)
    num_constraints = len(node_ids)
    for ct_index in range(num_constraints):
        num_nodes = len(node_ids[ct_index])
        if num_nodes == 2:
            points = []
            for node_index in range (num_nodes):
                x = na.node_x(details.node.blocks, node_ids[ct_index][node_index])
                points.append(x)
            segs.append(points)

    return segs
