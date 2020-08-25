"""
@author: Vincent Bonnet
@description : commands to use implementation of condition (condition_subclass)
"""

import logic
from lib.objects import Condition, Force

def add_wire_bending_constraint(scene, dynamic, stiffness, damping) -> Condition:
    condition = logic.WireBendingCondition([dynamic], stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_edge_constraint(scene, dynamic, stiffness, damping) -> Condition:
    condition = logic.EdgeCondition([dynamic], stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_face_constraint(scene, dynamic, stiffness, damping) -> Condition:
    condition = logic.AreaCondition([dynamic], stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_kinematic_attachment(scene, dynamic, kinematic, stiffness, damping, distance) -> Condition:
    condition = logic.KinematicAttachmentCondition(dynamic, kinematic, stiffness, damping, distance)
    scene.add_condition(condition)
    return condition

def add_dynamic_attachment(scene, dynamic0, dynamic1, stiffness, damping, distance) -> Condition:
    condition = logic.DynamicAttachmentCondition(dynamic0, dynamic1, stiffness, damping, distance)
    scene.add_condition(condition)
    return condition

def add_kinematic_collision(scene, dynamic, stiffness, damping) -> Condition:
    condition = logic.KinematicCollisionCondition(dynamic, stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_gravity(scene, gravity) -> Force:
    force = logic.Gravity(gravity)
    scene.add_force(force)
    return force
