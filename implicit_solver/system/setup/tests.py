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


def create_multi_wire_scene(context):
    '''
    Creates a scene with a wire attached to a kinematic object
    '''
    # wire
    wires = []
    for i in range(6):
        x = -2.0 + (i * 0.25)
        wire_shape = core.WireShape([x, 1.5], [x, -1.5] , WIRE_NUM_SEGMENTS)
        wire = objects.Dynamic(wire_shape, PARTICLE_MASS)
        wires.append(wire)

    # moving anchor and animation
    moving_anchor_shape = core.RectangleShape(min_x = -2.0, min_y = 1.5,
                                              max_x = 0.0, max_y =2.0)

    moving_anchor = objects.Kinematic(moving_anchor_shape)
    moving_anchor_position = moving_anchor.state.position
    func = lambda time: [[moving_anchor_position[0] + time,
                          moving_anchor_position[1]], 0.0]

    moving_anchor_animator = objects.Animator(func, context)

    # collider
    collider_shape = core.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 3,
                                       WIRE_ROOT_POS[0] + 0.5, WIRE_ROOT_POS[1] - 2)
    collider = objects.Kinematic(collider_shape)
    collider.rotation = 45

    scene = system.Scene()

    # Populate Scene with data and conditions
    for wire in wires:
        scene.addDynamic(wire)
        scene.addKinematic(moving_anchor, moving_anchor_animator)
        scene.addKinematic(collider)

        edge_condiction = utils.edge_constraint(scene, wire, stiffness=100.0, damping=0.0)
        wire_bending_condition = utils.wire_bending_constraint(scene, wire, stiffness=0.2, damping=0.0)
        utils.kinematic_attachment(scene, wire, moving_anchor, stiffness=100.0, damping=0.0, distance=0.1)
        utils.kinematic_collision(scene, wire, collider, stiffness=1000.0, damping=0.0)
        utils.gravity_acceleration(scene, GRAVITY)

        # Add Metadata to visualize the data and constraints
        utils.add_render_prefs(wire, ['co', 1])
        utils.add_render_prefs(edge_condiction, ['m-', 1])
        utils.add_render_prefs(wire_bending_condition, ['m-', 1])

    return scene

def create_wire_scene(context):
    '''
    Creates a scene with a wire attached to a kinematic object
    '''
    # wire
    wire_shape = core.WireShape(WIRE_ROOT_POS, WIRE_END_POS, WIRE_NUM_SEGMENTS)
    wire = objects.Dynamic(wire_shape, PARTICLE_MASS)

    # moving anchor and animation
    moving_anchor_shape = core.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 0.5,
                                              WIRE_ROOT_POS[0] + 0.25, WIRE_ROOT_POS[1])

    moving_anchor = objects.Kinematic(moving_anchor_shape)
    moving_anchor_position = moving_anchor.state.position
    decay_rate = 0.5
    func = lambda time: [[moving_anchor_position[0] + math.sin(time * 10.0) * math.pow(1.0-decay_rate, time),
                          moving_anchor_position[1]], math.sin(time * 10.0) * 90.0 * math.pow(1.0-decay_rate, time)]

    moving_anchor_animator = objects.Animator(func, context)

    # collider
    collider_shape = core.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 3,
                                       WIRE_ROOT_POS[0] + 0.5, WIRE_ROOT_POS[1] - 2)
    collider = objects.Kinematic(collider_shape)

    scene = system.Scene()

    # Populate Scene with data and conditions
    scene.addDynamic(wire)
    scene.addKinematic(moving_anchor, moving_anchor_animator)
    scene.addKinematic(collider)

    edge_condiction = utils.edge_constraint(scene, wire, stiffness=100.0, damping=0.0)
    wire_bending_condition = utils.wire_bending_constraint(scene, wire, stiffness=0.2, damping=0.0)
    utils.kinematic_attachment(scene, wire, moving_anchor, stiffness=100.0, damping=0.0, distance=0.1)
    utils.kinematic_collision(scene, wire, collider, stiffness=1000.0, damping=0.0)
    utils.gravity_acceleration(scene, GRAVITY)

    # Add Metadata
    utils.add_render_prefs(wire, ['co', 1])
    utils.add_render_prefs(edge_condiction, ['m-', 1])
    utils.add_render_prefs(wire_bending_condition, ['m-', 1])

    return scene

def create_beam_scene(context):
    '''
    Creates a scene with a beam and a wire
    '''
    # beam
    beam_shape = core.BeamShape(BEAM_POS, BEAM_WIDTH, BEAM_HEIGHT, BEAM_CELL_X, BEAM_CELL_Y)
    beam = objects.Dynamic(beam_shape, PARTICLE_MASS)

    # wire
    wire_start_pos = [BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT]
    wire_end_pos = [BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1] + BEAM_HEIGHT]
    wire_shape = core.WireShape(wire_start_pos, wire_end_pos, BEAM_CELL_X * 8)
    wire = objects.Dynamic(wire_shape, PARTICLE_MASS)

    # left anchor and animation
    left_anchor_shape = core.RectangleShape(BEAM_POS[0] - 0.5, BEAM_POS[1],
                                            BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT)
    left_anchor = objects.Kinematic(left_anchor_shape)
    l_pos = left_anchor.state.position
    func = lambda time: [[l_pos[0] + math.sin(2.0 * time) * 0.1, l_pos[1] + math.sin(time * 4.0)], 0.0]
    left_anchor_animator = objects.Animator(func, context)

    # right anchor and animation
    right_anchor_shape = core.RectangleShape(BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1],
                                             BEAM_POS[0] + BEAM_WIDTH + 0.5, BEAM_POS[1] + BEAM_HEIGHT)
    right_anchor = objects.Kinematic(right_anchor_shape)
    r_pos = right_anchor.state.position
    func = lambda time: [[r_pos[0] + math.sin(2.0 * time) * -0.1, r_pos[1]], 0.0]
    right_anchor_animator = objects.Animator(func, context)

    # Populate Scene with data and conditions
    scene = system.Scene()
    scene.addDynamic(beam)
    scene.addDynamic(wire)
    scene.addKinematic(left_anchor, left_anchor_animator)
    scene.addKinematic(right_anchor, right_anchor_animator)

    wire_edge_condition = utils.edge_constraint(scene, wire, stiffness=10.0, damping=0.0)
    beam_edge_condition = utils.edge_constraint(scene, beam, stiffness=20.0, damping=0.0)
    utils.face_constraint(scene, beam, stiffness=20.0, damping=0.0)
    utils.kinematic_attachment(scene, beam, right_anchor, stiffness=100.0, damping=0.0, distance=0.1)
    utils.kinematic_attachment(scene, beam, left_anchor, stiffness=100.0, damping=0.0, distance=0.1)
    utils.dynamic_attachment(scene, beam, wire, stiffness=100.0, damping=0.0, distance=0.001)
    utils.gravity_acceleration(scene, GRAVITY)

    # Add Metadata
    utils.add_render_prefs(beam, ['go', 1])
    utils.add_render_prefs(beam_edge_condition, ['k-', 1])

    utils.add_render_prefs(wire, ['co', 1])
    utils.add_render_prefs(wire_edge_condition, ['m-', 1])

    return scene
