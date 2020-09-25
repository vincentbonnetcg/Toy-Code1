"""
@author: Vincent Bonnet
@description : example scene with single wire
"""
import math
import lib.objects as objects
from lib.objects import BeamShape, WireShape, RectangleShape
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
    dispatcher.reset()
    context = dispatcher.get_context()
    # beam shape
    beam_shape = BeamShape(BEAM_POS, BEAM_WIDTH, BEAM_HEIGHT, BEAM_CELL_X, BEAM_CELL_Y)

    # wire shape
    wire_start_pos = [BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT]
    wire_end_pos = [BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1] + BEAM_HEIGHT]
    wire_shape = WireShape(wire_start_pos, wire_end_pos, BEAM_CELL_X * 8)

    # left anchor shape and animation
    l_anchor_shape = RectangleShape(BEAM_POS[0] - 0.5, BEAM_POS[1],
                                    BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT)
    l_pos, l_rot = l_anchor_shape.compute_best_transform()
    func = lambda time: [[l_pos[0] + math.sin(2.0 * time) * 0.1, l_pos[1] + math.sin(time * 4.0)], l_rot]
    l_animator = objects.Animator(func, context)

    # right anchor shape and animation
    r_anchor_shape = RectangleShape(BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1],
                                    BEAM_POS[0] + BEAM_WIDTH + 0.5, BEAM_POS[1] + BEAM_HEIGHT)
    r_pos, r_rot = r_anchor_shape.compute_best_transform()
    func = lambda time: [[r_pos[0] + math.sin(2.0 * time) * -0.1, r_pos[1]], r_rot]
    r_animator = objects.Animator(func, context)

    # Populate Scene with data and conditions
    dispatcher.add_dynamic(shape = beam_shape, node_mass = NODE_MASS, name = 'beam')
    dispatcher.add_dynamic(shape = wire_shape, node_mass = NODE_MASS, name = 'wire')

    dispatcher.add_kinematic(shape = l_anchor_shape, animator = l_animator, name = 'left_anchor')
    dispatcher.add_kinematic(shape = r_anchor_shape, animator = r_animator, name = 'right_anchor')

    dispatcher.add_edge_constraint(dynamic = 'beam', stiffness = 20.0, damping = 0.0, name = 'beam_edge')
    dispatcher.add_edge_constraint(dynamic = 'wire', stiffness = 10.0, damping = 0.0, name = 'wire_edge')
    dispatcher.add_face_constraint(dynamic = 'beam', stiffness = 20.0, damping = 0.0)

    dispatcher.add_kinematic_attachment(dynamic = 'beam', kinematic = 'left_anchor',
                                        stiffness = 100.0, damping = 0.0, distance = 0.1)

    dispatcher.add_kinematic_attachment(dynamic = 'beam', kinematic = 'right_anchor',
                                        stiffness = 100.0, damping = 0.0, distance = 0.1)

    dispatcher.add_dynamic_attachment(dynamic_0 = 'beam', dynamic_1 = 'wire',
                                      stiffness = 100.0, damping = 0.0, distance = 0.001)

    dispatcher.add_gravity(gravity = GRAVITY)

    # Set render preferences
    orange_color = common.meta_data_render(1.0, 'orange', 'solid', 0.8)
    grey_color = common.meta_data_render(1.0, 'grey', 'solid')
    blue_color = common.meta_data_render(1.0, 'blue', 'solid')
    green_color = common.meta_data_render(1.0, 'green', 'solid')

    dispatcher.set_render_prefs(obj = 'beam', prefs = grey_color)
    dispatcher.set_render_prefs(obj = 'beam_edge', prefs = blue_color)
    dispatcher.set_render_prefs(obj = 'wire', prefs = grey_color)
    dispatcher.set_render_prefs(obj = 'wire_edge', prefs = green_color)
    dispatcher.set_render_prefs(obj = 'left_anchor', prefs = orange_color)
    dispatcher.set_render_prefs(obj = 'right_anchor', prefs = orange_color)

    render.set_viewport_limit(-3.5, -3.5, 3.5, 3.5)
