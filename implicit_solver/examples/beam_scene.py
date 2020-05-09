"""
@author: Vincent Bonnet
@description : example scene with single wire
"""
import logic
import math
import lib.objects as objects
from . import common

BEAM_POS = [-4.0, 0.0] # in meters
BEAM_WIDTH = 8.0 # in meters
BEAM_HEIGHT = 1.0 # in meters
BEAM_CELL_X = 6 # number of cells along x
BEAM_CELL_Y = 4 # number of cells along y

NODE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2

def assemble(dispatcher, render):
    '''
    Initalizes a scene with a beam and a wire
    '''
    dispatcher.run('reset')
    context = dispatcher.run('get_context')
    # beam shape
    beam_shape = logic.BeamShape(BEAM_POS, BEAM_WIDTH, BEAM_HEIGHT, BEAM_CELL_X, BEAM_CELL_Y)

    # wire shape
    wire_start_pos = [BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT]
    wire_end_pos = [BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1] + BEAM_HEIGHT]
    wire_shape = logic.WireShape(wire_start_pos, wire_end_pos, BEAM_CELL_X * 8)

    # left anchor shape and animation
    left_anchor_shape = logic.RectangleShape(BEAM_POS[0] - 0.5, BEAM_POS[1],
                                            BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT)
    l_pos, l_rot = left_anchor_shape.compute_best_transform()
    func = lambda time: [[l_pos[0] + math.sin(2.0 * time) * 0.1, l_pos[1] + math.sin(time * 4.0)], l_rot]
    l_animator = objects.Animator(func, context)

    # right anchor shape and animation
    right_anchor_shape = logic.RectangleShape(BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1],
                                             BEAM_POS[0] + BEAM_WIDTH + 0.5, BEAM_POS[1] + BEAM_HEIGHT)
    r_pos, r_rot = right_anchor_shape.compute_best_transform()
    func = lambda time: [[r_pos[0] + math.sin(2.0 * time) * -0.1, r_pos[1]], r_rot]
    r_animator = objects.Animator(func, context)

    # Populate Scene with data and conditions
    beam_handle = dispatcher.run('add_dynamic', shape = beam_shape, node_mass = NODE_MASS)
    wire_handle = dispatcher.run('add_dynamic', shape = wire_shape, node_mass = NODE_MASS)

    left_anchor_handle = dispatcher.run('add_kinematic', shape = left_anchor_shape,
                                                         position = l_pos,
                                                         rotation = l_rot,
                                                         animator = l_animator)

    right_anchor_handle = dispatcher.run('add_kinematic', shape = right_anchor_shape,
                                                          position = r_pos,
                                                          rotation = r_rot,
                                                          animator = r_animator)

    beam_edge_condition_handle = dispatcher.run('add_edge_constraint', dynamic = beam_handle,
                                                                       stiffness = 20.0, damping = 0.0)

    wire_edge_condition_handle = dispatcher.run('add_edge_constraint', dynamic = wire_handle,
                                                                       stiffness = 10.0, damping = 0.0)

    dispatcher.run('add_face_constraint', dynamic = beam_handle,
                                           stiffness = 20.0, damping = 0.0)

    dispatcher.run('add_kinematic_attachment', dynamic = beam_handle, kinematic = left_anchor_handle,
                                               stiffness = 100.0, damping = 0.0, distance = 0.1)

    dispatcher.run('add_kinematic_attachment', dynamic = beam_handle, kinematic = right_anchor_handle,
                                               stiffness = 100.0, damping = 0.0, distance = 0.1)

    dispatcher.run('add_dynamic_attachment', dynamic0 = beam_handle, dynamic1 = wire_handle,
                                              stiffness = 100.0, damping = 0.0, distance = 0.001)

    dispatcher.run('add_gravity', gravity = GRAVITY)

    # Set render preferences
    dispatcher.run('set_render_prefs', obj = beam_handle,
                                       prefs = common.meta_data_render(1.0, 'grey', 'solid'))
    dispatcher.run('set_render_prefs', obj = beam_edge_condition_handle,
                                       prefs = common.meta_data_render(1.0, 'blue', 'solid'))

    dispatcher.run('set_render_prefs', obj = wire_handle,
                                       prefs = common.meta_data_render(1.0, 'grey', 'solid'))
    dispatcher.run('set_render_prefs', obj = wire_edge_condition_handle,
                                       prefs = common.meta_data_render(1.0, 'green', 'solid'))
    dispatcher.run('set_render_prefs', obj = left_anchor_handle,
                                       prefs = common.meta_data_render(1.0, 'orange', 'solid', 0.8))
    dispatcher.run('set_render_prefs', obj = right_anchor_handle,
                                       prefs = common.meta_data_render(1.0, 'orange', 'solid', 0.8))

    render.set_viewport_limit(-3.5, -3.5, 3.5, 3.5)

