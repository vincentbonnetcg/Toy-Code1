"""
@author: Vincent Bonnet
@description : utilities to setup simulation (scene, solver, context)
"""

import objects
import numpy as np

def add_render_prefs(dynamic, render_prefs):
    # Render preferences used by render.py
    # See : https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html for more details
    # fmt = '[color][marker][line]'
    # format of the display State ['fmt', size]
    dynamic.meta_data['render_prefs'] = render_prefs

def extract_transform_from_shape(shape):
    '''
    Returns the 'optimal' position and modify the shape vertices from world space to local space
    Optimal rotation is not computed
    '''
    optimal_pos = np.average(shape.vertex.position, axis=0)
    np.subtract(shape.vertex.position, optimal_pos, out=shape.vertex.position)
    optimal_rot = 0
    return optimal_pos, optimal_rot

def add_kinematic(scene, shape, position = (0., 0.), rotation = 0., animator = None):
    kinematic = objects.Kinematic(shape, position, rotation)
    scene.add_kinematic(kinematic, animator)
    return kinematic

def add_dynamic(scene, shape, particle_mass):
    dynamic = objects.Dynamic(shape, particle_mass)
    scene.add_dynamic(dynamic)
    return dynamic

def add_wire_bending_constraint(scene, dynamic, stiffness, damping):
    condition = objects.WireBendingCondition([dynamic], stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_edge_constraint(scene, dynamic, stiffness, damping):
    condition = objects.SpringCondition([dynamic], stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_face_constraint(scene, dynamic, stiffness, damping):
    condition = objects.AreaCondition([dynamic], stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_kinematic_attachment(scene, dynamic, kinematic, stiffness, damping, distance):
    condition = objects.KinematicAttachmentCondition(dynamic, kinematic, stiffness, damping, distance)
    scene.add_condition(condition)
    return condition

def add_dynamic_attachment(scene, dynamic0, dynamic1, stiffness, damping, distance):
    condition = objects.DynamicAttachmentCondition(dynamic0, dynamic1, stiffness, damping, distance)
    scene.add_condition(condition)
    return condition

def add_kinematic_collision(scene, dynamic, kinematic, stiffness, damping):
    condition = objects.KinematicCollisionCondition(dynamic, kinematic, stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_gravity_acceleration(scene, gravity):
    force = objects.Gravity(gravity)
    scene.add_force(force)
    return force