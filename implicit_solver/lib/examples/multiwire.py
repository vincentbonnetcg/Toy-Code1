"""
@author: Vincent Bonnet
@description : example scene with multiwire
"""

import lib.objects as objects
from . import common
from lib.objects import WireShape, RectangleShape

WIRE_ROOT_POS = [0.0, 2.0] # in meters
WIRE_END_POS = [0.0, -2.0] # in meters
WIRE_NUM_SEGMENTS = 30

NODE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2

def assemble(dispatcher, render):
    '''
    Initalizes a scene with multiple wire attached to a kinematic object
    '''
    dispatcher.reset()
    context = dispatcher.get_context()

    orange_color = common.meta_data_render(1.0, 'orange', 'solid', 0.8)
    blue_color = common.meta_data_render(1.0, 'blue', 'solid')
    green_color = common.meta_data_render(1.0, 'green', 'solid')

    # wire shape
    wire_shapes = []
    for i in range(6):
        x = -2.0 + (i * 0.25)
        wire_shape = WireShape([x, 1.5], [x, -1.5] , WIRE_NUM_SEGMENTS)
        wire_shapes.append(wire_shape)

    # anchor shape and animation
    anchor_shape = RectangleShape(min_x = -2.0, min_y = 1.5,
                                              max_x = 0.0, max_y =2.0)
    anchor_position, anchor_rotation = anchor_shape.compute_best_transform()
    func = lambda time: [[anchor_position[0] + time,
                          anchor_position[1]], 0.0]

    anchor_animator = objects.Animator(func, context)

    # collider shape
    collider_shape = RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 3,
                                    WIRE_ROOT_POS[0] + 0.5, WIRE_ROOT_POS[1] - 2)
    collider_position, collider_rotation = anchor_shape.compute_best_transform()
    collider_shape.transform(collider_position, 45.0)

    # Populate Scene with data and conditions
    dispatcher.add_kinematic(shape = anchor_shape, animator = anchor_animator, name = 'anchor')
    dispatcher.add_kinematic(shape = collider_shape, name = 'collider')

    for wire_id, wire_shape in enumerate(wire_shapes):
        wire_name = f'wire{wire_id}'
        wire_edge_name = f'wire_edge{wire_id}'

        dispatcher.add_dynamic(shape = wire_shape, node_mass = NODE_MASS, name = wire_name)

        dispatcher.add_edge_constraint(dynamic = wire_name, stiffness = 100.0,
                                       damping = 0.0, name = wire_edge_name)

        dispatcher.add_wire_bending_constraint(dynamic= wire_name,
                                               stiffness = 0.15, damping = 0.0)

        dispatcher.add_kinematic_attachment(dynamic = wire_name, kinematic = 'anchor',
                                            stiffness = 100.0, damping = 0.0, distance = 0.1)

        dispatcher.set_render_prefs(obj = wire_name, prefs = blue_color)
        dispatcher.set_render_prefs(obj = wire_edge_name, prefs = green_color)

    dispatcher.add_kinematic_collision(stiffness = 10000.0, damping = 0.0)

    dispatcher.add_gravity(gravity = GRAVITY)

    # Set render preferences
    dispatcher.set_render_prefs(obj = 'anchor', prefs = orange_color)
    dispatcher.set_render_prefs(obj = 'collider', prefs = orange_color)

    render.set_viewport_limit(-2.5, -2.5, 2.5, 2.5)

