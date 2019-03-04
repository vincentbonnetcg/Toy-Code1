"""
@author: Vincent Bonnet
@description : commands to setup objects and run simulation
"""

import objects

def set_render_prefs(obj, prefs):
    '''
    Render preferences used by render.py
    See : https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html for more details
    fmt = '[color][marker][line]'
    format of the display State ['fmt', size]
    '''
    obj.meta_data['render_prefs'] = prefs

def add_kinematic(scene, shape, position = (0., 0.), rotation = 0., animator = None):
    '''
    Add a Kinematic object
    '''
    kinematic = objects.Kinematic(shape, position, rotation)
    scene.add_kinematic(kinematic, animator)
    return kinematic

def add_dynamic(scene, shape, particle_mass):
    '''
    Add a Dynamic object
    '''
    dynamic = objects.Dynamic(shape, particle_mass)
    scene.add_dynamic(dynamic)
    return dynamic

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
