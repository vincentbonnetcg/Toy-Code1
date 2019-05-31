"""
@author: Vincent Bonnet
@description : commands to use implementation of condition (condition_subclass)
"""

import tests
from objects import Condition, Force

def add_wire_bending_constraint(scene, dynamic, stiffness, damping) -> Condition:
    condition = tests.WireBendingCondition([dynamic], stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_edge_constraint(scene, dynamic, stiffness, damping) -> Condition:
    condition = tests.EdgeCondition.create([dynamic], stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_face_constraint(scene, dynamic, stiffness, damping) -> Condition:
    condition = tests.AreaCondition([dynamic], stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_kinematic_attachment(scene, dynamic, kinematic, stiffness, damping, distance) -> Condition:
    condition = tests.KinematicAttachmentCondition(dynamic, kinematic, stiffness, damping, distance)
    scene.add_condition(condition)
    return condition

def add_dynamic_attachment(scene, dynamic0, dynamic1, stiffness, damping, distance) -> Condition:
    condition = tests.DynamicAttachmentCondition(dynamic0, dynamic1, stiffness, damping, distance)
    scene.add_condition(condition)
    return condition

def add_kinematic_collision(scene, dynamic, kinematic, stiffness, damping) -> Condition:
    condition = tests.KinematicCollisionCondition(dynamic, kinematic, stiffness, damping)
    scene.add_condition(condition)
    return condition

def add_gravity(scene, gravity) -> Force:
    force = tests.Gravity(gravity)
    scene.add_force(force)
    return force
