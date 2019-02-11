"""
@author: Vincent Bonnet
@description : example scenes for unit testing
"""

import objects
import system
import core
import math
import system.setup.utilities as utils


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

PARTICLE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2


def init_multi_wire_scene(scene, context):
    '''
    Initalizes a scene with a wire attached to a kinematic object
    '''
    # wire shape
    wire_shapes = []
    for i in range(6):
        x = -2.0 + (i * 0.25)
        wire_shape = core.WireShape([x, 1.5], [x, -1.5] , WIRE_NUM_SEGMENTS)
        wire_shapes.append(wire_shape)

    # anchor shape and animation
    moving_anchor_shape = core.RectangleShape(min_x = -2.0, min_y = 1.5,
                                              max_x = 0.0, max_y =2.0)
    moving_anchor_position, moving_anchor_rotation = utils.extract_transform_from_shape(moving_anchor_shape)
    func = lambda time: [[moving_anchor_position[0] + time,
                          moving_anchor_position[1]], 0.0]

    moving_anchor_animator = objects.Animator(func, context)

    # collider shape
    collider_shape = core.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 3,
                                       WIRE_ROOT_POS[0] + 0.5, WIRE_ROOT_POS[1] - 2)
    collider_position, collider_rotation = utils.extract_transform_from_shape(moving_anchor_shape)
    collider_rotation = 45.

    # Populate Scene with data and conditions
    moving_anchor = utils.add_kinematic(scene, moving_anchor_shape,
                                                moving_anchor_position,
                                                moving_anchor_rotation,
                                                moving_anchor_animator)

    collider = utils.add_kinematic(scene, collider_shape,
                                           collider_position,
                                           collider_rotation)

    for wire_shape in wire_shapes:
        wire = utils.add_dynamic(scene, wire_shape, PARTICLE_MASS)
        edge_condiction = utils.add_edge_constraint(scene, wire, stiffness=100.0, damping=0.0)
        wire_bending_condition = utils.add_wire_bending_constraint(scene, wire, stiffness=0.2, damping=0.0)
        utils.add_kinematic_attachment(scene, wire, moving_anchor, stiffness=100.0, damping=0.0, distance=0.1)
        utils.add_kinematic_collision(scene, wire, collider, stiffness=1000.0, damping=0.0)
        utils.add_gravity_acceleration(scene, GRAVITY)

        # Add Metadata to visualize the data and constraints
        utils.add_render_prefs(wire, ['co', 1])
        utils.add_render_prefs(edge_condiction, ['m-', 1])
        utils.add_render_prefs(wire_bending_condition, ['m-', 1])

    return scene

def init_wire_scene(scene, context):
    '''
    Initalizes a scene with a wire attached to a kinematic object
    '''
    # wire shape
    wire_shape = core.WireShape(WIRE_ROOT_POS, WIRE_END_POS, WIRE_NUM_SEGMENTS)

    # collider shape
    collider_shape = core.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 3,
                                       WIRE_ROOT_POS[0] + 0.5, WIRE_ROOT_POS[1] - 2)

    # anchor shape and animation
    moving_anchor_shape = core.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 0.5,
                                              WIRE_ROOT_POS[0] + 0.25, WIRE_ROOT_POS[1])

    moving_anchor_position, moving_anchor_rotation = utils.extract_transform_from_shape(moving_anchor_shape)
    decay_rate = 0.5
    func = lambda time: [[moving_anchor_position[0] + math.sin(time * 10.0) * math.pow(1.0-decay_rate, time),
                          moving_anchor_position[1]], math.sin(time * 10.0) * 90.0 * math.pow(1.0-decay_rate, time)]
    moving_anchor_animator = objects.Animator(func, context)

    # Populate Scene with data and conditions
    wire = utils.add_dynamic(scene, wire_shape, PARTICLE_MASS)
    collider = utils.add_kinematic(scene, collider_shape)
    moving_anchor = utils.add_kinematic(scene, moving_anchor_shape,
                                                moving_anchor_position,
                                                moving_anchor_rotation,
                                                moving_anchor_animator)

    edge_condiction = utils.add_edge_constraint(scene, wire, stiffness=100.0, damping=0.0)
    wire_bending_condition = utils.add_wire_bending_constraint(scene, wire, stiffness=0.2, damping=0.0)
    utils.add_kinematic_attachment(scene, wire, moving_anchor, stiffness=100.0, damping=0.0, distance=0.1)
    utils.add_kinematic_collision(scene, wire, collider, stiffness=1000.0, damping=0.0)
    utils.add_gravity_acceleration(scene, GRAVITY)

    # Add Metadata
    utils.add_render_prefs(wire, ['co', 1])
    utils.add_render_prefs(edge_condiction, ['m-', 1])
    utils.add_render_prefs(wire_bending_condition, ['m-', 1])

    return scene

def init_beam_scene(scene, context):
    '''
    Initalizes a scene with a beam and a wire
    '''
    # beam shape
    beam_shape = core.BeamShape(BEAM_POS, BEAM_WIDTH, BEAM_HEIGHT, BEAM_CELL_X, BEAM_CELL_Y)

    # wire shape
    wire_start_pos = [BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT]
    wire_end_pos = [BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1] + BEAM_HEIGHT]
    wire_shape = core.WireShape(wire_start_pos, wire_end_pos, BEAM_CELL_X * 8)

    # left anchor shape and animation
    left_anchor_shape = core.RectangleShape(BEAM_POS[0] - 0.5, BEAM_POS[1],
                                            BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT)
    l_pos, l_rot = utils.extract_transform_from_shape(left_anchor_shape)
    func = lambda time: [[l_pos[0] + math.sin(2.0 * time) * 0.1, l_pos[1] + math.sin(time * 4.0)], l_rot]
    l_animator = objects.Animator(func, context)

    # right anchor shape and animation
    right_anchor_shape = core.RectangleShape(BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1],
                                             BEAM_POS[0] + BEAM_WIDTH + 0.5, BEAM_POS[1] + BEAM_HEIGHT)
    r_pos, r_rot = utils.extract_transform_from_shape(right_anchor_shape)
    func = lambda time: [[r_pos[0] + math.sin(2.0 * time) * -0.1, r_pos[1]], r_rot]
    r_animator = objects.Animator(func, context)

    # Populate Scene with data and conditions
    beam = utils.add_dynamic(scene, beam_shape, PARTICLE_MASS)
    wire = utils.add_dynamic(scene, wire_shape, PARTICLE_MASS)
    left_anchor = utils.add_kinematic(scene, left_anchor_shape, l_pos, l_rot, l_animator)
    right_anchor = utils.add_kinematic(scene, right_anchor_shape, r_pos, r_rot, r_animator)

    wire_edge_condition = utils.add_edge_constraint(scene, wire, stiffness=10.0, damping=0.0)
    beam_edge_condition = utils.add_edge_constraint(scene, beam, stiffness=20.0, damping=0.0)
    utils.add_face_constraint(scene, beam, stiffness=20.0, damping=0.0)
    utils.add_kinematic_attachment(scene, beam, right_anchor, stiffness=100.0, damping=0.0, distance=0.1)
    utils.add_kinematic_attachment(scene, beam, left_anchor, stiffness=100.0, damping=0.0, distance=0.1)
    utils.add_dynamic_attachment(scene, beam, wire, stiffness=100.0, damping=0.0, distance=0.001)
    utils.add_gravity_acceleration(scene, GRAVITY)

    # Add Metadata
    utils.add_render_prefs(beam, ['go', 1])
    utils.add_render_prefs(beam_edge_condition, ['k-', 1])

    utils.add_render_prefs(wire, ['co', 1])
    utils.add_render_prefs(wire_edge_condition, ['m-', 1])

    return scene
