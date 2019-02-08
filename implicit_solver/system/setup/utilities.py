"""
@author: Vincent Bonnet
@description : utilities to setup simulation (scene, solver, context)
"""

import objects

def add_render_prefs(dynamic, render_prefs):
    # Render preferences used by render.py
    # See : https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html for more details
    # fmt = '[color][marker][line]'
    # format of the display State ['fmt', size]
    dynamic.meta_data['render_prefs'] = render_prefs

def wire_bending_constraint(scene, dynamic, stiffness, damping):
    condition = objects.WireBendingCondition([dynamic], stiffness, damping)
    scene.addCondition(condition)
    return condition

def edge_constraint(scene, dynamic, stiffness, damping):
    condition = objects.SpringCondition([dynamic], stiffness, damping)
    scene.addCondition(condition)
    return condition

def face_constraint(scene, dynamic, stiffness, damping):
    condition = objects.AreaCondition([dynamic], stiffness, damping)
    scene.addCondition(condition)
    return condition

def kinematic_attachment(scene, dynamic, kinematic, stiffness, damping, distance):
    condition = objects.KinematicAttachmentCondition(dynamic, kinematic, stiffness, damping, distance)
    scene.addCondition(condition)
    return condition

def dynamic_attachment(scene, dynamic0, dynamic1, stiffness, damping, distance):
    condition = objects.DynamicAttachmentCondition(dynamic0, dynamic1, stiffness, damping, distance)
    scene.addCondition(condition)
    return condition

def kinematic_collision(scene, dynamic, kinematic, stiffness, damping):
    condition = objects.KinematicCollisionCondition(dynamic, kinematic, stiffness, damping)
    scene.addCondition(condition)
    return condition

def gravity_acceleration(scene, gravity):
    force = objects.Gravity(gravity)
    scene.addForce(force)
    return force