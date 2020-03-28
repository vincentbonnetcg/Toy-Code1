"""
@author: Vincent Bonnet
@description : commands to setup objects and run simulation
"""

import lib.objects as objects
import lib.common as cm
import lib.common.jit.node_accessor as na

def set_render_prefs(obj, prefs):
    '''
    Render preferences used by render.py
    '''
    obj.meta_data['render_prefs'] = prefs

def add_kinematic(scene, details, shape, position = (0., 0.), rotation = 0., animator = None):
    '''
    Add a Kinematic object
    '''
    kinematic = objects.Kinematic(details, shape, position, rotation)
    scene.add_kinematic(kinematic, animator)
    return kinematic

def add_dynamic(scene, details, shape, node_mass):
    '''
    Add a Dynamic object
    '''
    dynamic = objects.Dynamic(details, shape, node_mass)
    scene.add_dynamic(dynamic)
    return dynamic

@cm.timeit
def solve_to_next_frame(scene, solver, details, context):
    '''
    Solve the scene and move to the next frame
    '''
    for _ in range(context.num_substep):
        context.time += context.dt
        solver.solve_step(scene, details, context)

def initialize(scene, solver, details, context):
    '''
    Initialize the solver
    '''
    solver.initialize(scene, details, context)

def get_nodes_from_dynamic(scene, index, details):
    '''
    Get node position from dynamic object
    '''
    dynamic = scene.dynamics[index]
    return details.node.flatten('x', dynamic.block_handles)

def get_shape_from_kinematic(scene, index, details):
    '''
    Get points from kinematic
    '''
    kinematic = scene.kinematics[index]
    return kinematic.get_as_shape(details)

def get_segments_from_constraint(scene, index, details):
    '''
    Get position from constraint object
    '''
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
