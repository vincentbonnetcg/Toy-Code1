"""
@author: Vincent Bonnet
@description : commands to setup objects and run simulation
"""

import lib.objects as objects
import lib.common as cm

def set_render_prefs(obj, prefs):
    '''
    Render preferences used by render.py
    '''
    obj.meta_data['render_prefs'] = prefs

def add_kinematic(scene, shape, position = (0., 0.), rotation = 0., animator = None):
    '''
    Add a Kinematic object
    '''
    kinematic = objects.Kinematic(shape, position, rotation)
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

def get_position_from_dynamic(scene, index, details):
    '''
    Get channel from dynamic object
    '''
    dynamic = scene.dynamics[index]
    return details.node.flatten('x', dynamic.blocks_ids)
