"""
@author: Vincent Bonnet
@description : example scene with single wire
"""
import logic
import math
import lib.objects as objects
from . import common

WIRE_ROOT_POS = [0.0, 2.0] # in meters
WIRE_END_POS = [0.0, -2.0] # in meters
WIRE_NUM_SEGMENTS = 30

NODE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2

def assemble(dispatcher, render):
    '''
    Initalizes a scene with a wire attached to a kinematic object
    '''
    dispatcher.run('reset')
    context = dispatcher.run('get_context')
    # wire shape
    wire_shape = logic.WireShape(WIRE_ROOT_POS, WIRE_END_POS, WIRE_NUM_SEGMENTS)

    # collider shape
    collider_shape = logic.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 3.5,
                                    WIRE_ROOT_POS[0] + 0.5, WIRE_ROOT_POS[1] - 2)

    # anchor shape and animation
    moving_anchor_shape = logic.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 0.5,
                                              WIRE_ROOT_POS[0] + 0.25, WIRE_ROOT_POS[1])

    moving_anchor_position, moving_anchor_rotation = moving_anchor_shape.compute_best_transform()
    decay_rate = 0.5
    func = lambda time: [[moving_anchor_position[0] + math.sin(time * 10.0) * math.pow(1.0-decay_rate, time),
                          moving_anchor_position[1]], math.sin(time * 10.0) * 90.0 * math.pow(1.0-decay_rate, time)]
    moving_anchor_animator = objects.Animator(func, context)

    # Populate scene with commands
    wire_handle = dispatcher.run('add_dynamic', shape = wire_shape, node_mass = NODE_MASS)
    collider_handle = dispatcher.run('add_kinematic', shape = collider_shape)

    moving_anchor_handle = dispatcher.run('add_kinematic', shape = moving_anchor_shape,
                                                          position = moving_anchor_position,
                                                          rotation = moving_anchor_rotation,
                                                          animator =moving_anchor_animator)

    edge_condition_handle = dispatcher.run('add_edge_constraint', dynamic = wire_handle,
                                                                   stiffness = 100.0, damping = 0.0)
    dispatcher.run('add_wire_bending_constraint', dynamic= wire_handle,
                                                   stiffness = 0.2, damping = 0.0)
    dispatcher.run('add_kinematic_attachment', dynamic = wire_handle, kinematic = moving_anchor_handle,
                                               stiffness = 100.0, damping = 0.0, distance = 0.1)
    dispatcher.run('add_kinematic_collision', dynamic = wire_handle, kinematic = collider_handle,
                                               stiffness = 1000.0, damping = 0.0)
    dispatcher.run('add_gravity', gravity = GRAVITY)

    # Set render preferences
    dispatcher.run('set_render_prefs', obj = wire_handle,
                                       prefs = common.meta_data_render(1.0, 'blue', 'solid'))
    dispatcher.run('set_render_prefs', obj = edge_condition_handle,
                                       prefs = common.meta_data_render(1.0, 'green', 'solid'))
    dispatcher.run('set_render_prefs', obj = collider_handle,
                                       prefs = common.meta_data_render(1.0, 'orange', 'solid', 0.8))
    dispatcher.run('set_render_prefs', obj = moving_anchor_handle,
                                       prefs = common.meta_data_render(1.0, 'orange', 'solid', 0.8))

    render.set_viewport_limit(-2.5, -2.5, 2.5, 2.5)