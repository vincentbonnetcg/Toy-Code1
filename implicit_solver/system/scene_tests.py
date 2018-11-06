"""
@author: Vincent Bonnet
@description : example scenes for unit testing
"""

import math
import objects
import system
import constraints as cn

'''
 Global Constants
'''
WIRE_ROOT_POS = [0.0, 2.0] # in meters
WIRE_END_POS = [0.0, -2.0] # in meters
WIRE_NUM_SEGMENTS = 30

BEAM_POS = [-4.0, 0.0] # in meters
BEAM_WIDTH = 8.0 # in meters
BEAM_HEIGHT = 1.0 # in meters
BEAM_CELL_X = 6 # number of cells along x
BEAM_CELL_Y = 4 # number of cells along y

STIFFNESS = 2.0 # in newtons per meter (N/m)
DAMPING = 0.0
PARTICLE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2

def add_render_prefs(dynamic, render_prefs):
    # Render preferences used by render.py
    # See : https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.plot.html for more details
    # fmt = '[color][marker][line]'
    # format of the display State ['fmt', size]
    dynamic.meta_data['render_prefs'] = render_prefs

def wire_bending_constraint(scene, dynamic, stiffness, damping):
    constraint_builder = cn.WireBendingBuilder([dynamic], stiffness, damping)
    scene.addConstraintBuilder(constraint_builder)
    return constraint_builder

def edge_constraint(scene, dynamic, stiffness, damping):
    constraint_builder = cn.SpringBuilder([dynamic], stiffness, damping)
    scene.addConstraintBuilder(constraint_builder)
    return constraint_builder

def face_constraint(scene, dynamic, stiffness, damping):
    constraint_builder = cn.AreaBuilder([dynamic], stiffness, damping)
    scene.addConstraintBuilder(constraint_builder)
    return constraint_builder

def kinematic_attachment(scene, dynamic, kinematic, stiffness, damping, distance):
    attachment_builder = cn.KinematicAttachmentBuilder(dynamic, kinematic, stiffness, damping, distance)
    scene.addConstraintBuilder(attachment_builder)
    return attachment_builder

def dynamic_attachment(scene, dynamic0, dynamic1, stiffness, damping, distance):
    attachment_builder = cn.DynamicAttachmentBuilder(dynamic0, dynamic1, stiffness, damping, distance)
    scene.addConstraintBuilder(attachment_builder)
    return attachment_builder

def kinematic_collision(scene, dynamic, kinematic, stiffness, damping):
    collison_builder = cn.KinematicCollisionBuilder(dynamic, kinematic, stiffness, damping)
    scene.addConstraintBuilder(collison_builder)
    return collison_builder

def create_wire_scene():
    '''
    Creates a scene with a wire attached to a kinematic object
    '''
    wire_shape = objects.WireShape(WIRE_ROOT_POS, WIRE_END_POS, WIRE_NUM_SEGMENTS)
    wire = objects.Dynamic(wire_shape, PARTICLE_MASS)

    moving_anchor = objects.Rectangle(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 0.5, WIRE_ROOT_POS[0] + 0.25, WIRE_ROOT_POS[1])
    moving_anchor_position = moving_anchor.position
    decay_rate = 0.6
    moving_anchor_animation = lambda time: [[moving_anchor_position[0] + math.sin(time * 10.0) * math.pow(1.0-decay_rate, time),
                                             moving_anchor_position[1]], math.sin(time * 10.0) * 90.0 * math.pow(1.0-decay_rate, time)]
    moving_anchor.animationFunc = moving_anchor_animation

    collider = objects.Rectangle(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 3, WIRE_ROOT_POS[0] + 0.5, WIRE_ROOT_POS[1] - 2)

    scene = system.Scene(GRAVITY)

    # Populate Scene with data and conditions
    scene.addDynamic(wire)
    scene.addKinematic(moving_anchor)
    scene.addKinematic(collider)

    edge_condiction = edge_constraint(scene, wire, STIFFNESS * 50.0, DAMPING)
    wire_bending_condition = wire_bending_constraint(scene, wire, STIFFNESS * 0.1, DAMPING)
    kinematic_attachment(scene, wire, moving_anchor, 100.0, 0.0, 0.1)
    kinematic_collision(scene, wire, collider, 1000.0, 0.0)

    # Add Metadata
    add_render_prefs(wire, ['co', 0])
    add_render_prefs(edge_condiction, ['m-', 1])
    add_render_prefs(wire_bending_condition, ['m-', 1])

    return scene

def create_beam_scene():
    '''
    Creates a scene with a beam and a wire
    '''
    beam_shape = objects.BeamShape(BEAM_POS, BEAM_WIDTH, BEAM_HEIGHT, BEAM_CELL_X, BEAM_CELL_Y)
    beam = objects.Dynamic(beam_shape, PARTICLE_MASS)

    wire_start_pos = [BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT]
    wire_end_pos = [BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1] + BEAM_HEIGHT]
    wire_shape = objects.WireShape(wire_start_pos, wire_end_pos, BEAM_CELL_X * 8)
    wire = objects.Dynamic(wire_shape, PARTICLE_MASS * 0.1)

    left_anchor = objects.Rectangle(BEAM_POS[0] - 0.5, BEAM_POS[1], BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT)
    l_pos = left_anchor.position
    left_anchor.animationFunc = lambda time: [[l_pos[0] + math.sin(2.0 * time) * 0.1, l_pos[1] + math.sin(time * 4.0)], 0.0]

    right_anchor = objects.Rectangle(BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1], BEAM_POS[0] + BEAM_WIDTH + 0.5, BEAM_POS[1] + BEAM_HEIGHT)
    r_pos = right_anchor.position
    right_anchor.animationFunc = lambda time: [[r_pos[0] + math.sin(2.0 * time) * -0.1, r_pos[1]], 0.0]

    # Populate Scene with data and conditions
    scene = system.Scene(GRAVITY)
    scene.addDynamic(beam)
    scene.addDynamic(wire)
    scene.addKinematic(left_anchor)
    scene.addKinematic(right_anchor)

    wire_edge_condition = edge_constraint(scene, wire, STIFFNESS * 0.5, DAMPING)
    beam_edge_condition = edge_constraint(scene, beam, STIFFNESS * 10.0, DAMPING)
    face_constraint(scene, beam, STIFFNESS * 10.0, DAMPING)
    kinematic_attachment(scene, beam, right_anchor, 100.0, 0.0, 0.1)
    kinematic_attachment(scene, beam, left_anchor, 100.0, 0.0, 0.1)
    dynamic_attachment(scene, beam, wire, 100.0, 0.0, 0.001)

    # Add Metadata
    add_render_prefs(beam, ['go', 1])
    add_render_prefs(beam_edge_condition, ['k-', 1])

    add_render_prefs(wire, ['co', 1])
    add_render_prefs(wire_edge_condition, ['m-', 1])

    return scene
