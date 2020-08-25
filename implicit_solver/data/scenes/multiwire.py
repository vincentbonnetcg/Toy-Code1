"""
@author: Vincent Bonnet
@description : example scene with multiwire
"""
import logic
import lib.objects as objects
from . import common

WIRE_ROOT_POS = [0.0, 2.0] # in meters
WIRE_END_POS = [0.0, -2.0] # in meters
WIRE_NUM_SEGMENTS = 30

NODE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2

def assemble(dispatcher, render):
    '''
    Initalizes a scene with multiple wire attached to a kinematic object
    '''
    dispatcher.run('reset')
    context = dispatcher.run('get_context')
    # wire shape
    wire_shapes = []
    for i in range(6):
        x = -2.0 + (i * 0.25)
        wire_shape = logic.WireShape([x, 1.5], [x, -1.5] , WIRE_NUM_SEGMENTS)
        wire_shapes.append(wire_shape)

    # anchor shape and animation
    moving_anchor_shape = logic.RectangleShape(min_x = -2.0, min_y = 1.5,
                                              max_x = 0.0, max_y =2.0)
    moving_anchor_position, moving_anchor_rotation = moving_anchor_shape.compute_best_transform()
    func = lambda time: [[moving_anchor_position[0] + time,
                          moving_anchor_position[1]], 0.0]

    moving_anchor_animator = objects.Animator(func, context)

    # collider shape
    collider_shape = logic.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 3,
                                       WIRE_ROOT_POS[0] + 0.5, WIRE_ROOT_POS[1] - 2)
    collider_position, collider_rotation = moving_anchor_shape.compute_best_transform()
    collider_rotation = 45.0

    # Populate Scene with data and conditions
    moving_anchor_handle = dispatcher.run('add_kinematic', shape = moving_anchor_shape,
                                                          position = moving_anchor_position,
                                                          rotation = moving_anchor_rotation,
                                                          animator =moving_anchor_animator)

    collider_handle = dispatcher.run('add_kinematic', shape = collider_shape,
                                                        position = collider_position,
                                                        rotation = collider_rotation)

    for wire_shape in wire_shapes:
        wire_handle = dispatcher.run('add_dynamic', shape = wire_shape, node_mass = NODE_MASS)

        edge_condition_handle = dispatcher.run('add_edge_constraint', dynamic = wire_handle,
                                                               stiffness = 100.0, damping = 0.0)

        dispatcher.run('add_wire_bending_constraint', dynamic= wire_handle,
                                                       stiffness = 0.15, damping = 0.0)

        dispatcher.run('add_kinematic_attachment', dynamic = wire_handle, kinematic = moving_anchor_handle,
                                                   stiffness = 100.0, damping = 0.0, distance = 0.1)

        dispatcher.run('add_kinematic_collision', dynamic = wire_handle, stiffness = 10000.0, damping = 0.0)

        dispatcher.run('set_render_prefs', obj = wire_handle,
                                           prefs = common.meta_data_render(1.0, 'blue', 'solid'))
        dispatcher.run('set_render_prefs', obj = edge_condition_handle,
                                           prefs = common.meta_data_render(1.0, 'green', 'solid'))

    dispatcher.run('add_gravity', gravity = GRAVITY)

    # Set render preferences
    dispatcher.run('set_render_prefs', obj = moving_anchor_handle,
                                       prefs = common.meta_data_render(1.0, 'orange', 'solid', 0.8))
    dispatcher.run('set_render_prefs', obj = collider_handle,
                                       prefs = common.meta_data_render(1.0, 'orange', 'solid', 0.8))

    render.set_viewport_limit(-2.5, -2.5, 2.5, 2.5)

