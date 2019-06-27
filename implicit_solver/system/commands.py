"""
@author: Vincent Bonnet
@description : commands to setup objects and run simulation
"""

import objects
from core import profiler

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

def add_dynamic(scene, shape, node_mass):
    '''
    Add a Dynamic object
    '''
    dynamic = objects.Dynamic(shape, node_mass)
    scene.add_dynamic(dynamic)
    return dynamic

@profiler.timeit
def solve_to_next_frame(scene, solver, context):
    '''
    Solve the scene and move to the next frame
    '''
    for _ in range(context.num_substep):
        context.time += context.dt
        solver.solve_step(scene, context)

def initialize(scene, solver, context):
    '''
    Initialize the solver
    '''
    solver.initialize(scene, context)
