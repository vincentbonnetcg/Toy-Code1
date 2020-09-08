"""
@author: Vincent Bonnet
@description : example scene with single wire
"""
import math
import lib.objects as objects
import lib.objects.logic as logic
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
    dispatcher.reset()
    context = dispatcher.get_context()
    # wire shape
    wire_shape = logic.WireShape(WIRE_ROOT_POS, WIRE_END_POS, WIRE_NUM_SEGMENTS)

    # collider shape
    collider_shape = logic.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 3.5,
                                    WIRE_ROOT_POS[0] + 0.5, WIRE_ROOT_POS[1] - 2)

    # anchor shape and animation
    anchor_shape = logic.RectangleShape(WIRE_ROOT_POS[0], WIRE_ROOT_POS[1] - 0.5,
                                              WIRE_ROOT_POS[0] + 0.25, WIRE_ROOT_POS[1])

    anchor_position, anchor_rotation = anchor_shape.compute_best_transform()
    decay_rate = 0.5
    func = lambda time: [[anchor_position[0] + math.sin(time * 10.0) * math.pow(1.0-decay_rate, time),
                          anchor_position[1]], math.sin(time * 10.0) * 90.0 * math.pow(1.0-decay_rate, time)]
    anchor_animator = objects.Animator(func, context)

    # Populate scene
    dispatcher.add_dynamic(shape = wire_shape, node_mass = NODE_MASS, name='wire')
    dispatcher.add_kinematic(shape = collider_shape, name='collider')
    dispatcher.add_kinematic(shape = anchor_shape, animator = anchor_animator, name='anchor')
    dispatcher.add_edge_constraint(dynamic = 'wire', stiffness = 100.0, damping = 0.0, name='edge_constraint')
    dispatcher.add_wire_bending_constraint(dynamic= 'wire', stiffness = 0.2, damping = 0.0)
    dispatcher.add_kinematic_attachment(dynamic = 'wire', kinematic = 'anchor',
                                               stiffness = 100.0, damping = 0.0, distance = 0.1)
    dispatcher.add_kinematic_collision(stiffness = 1000.0, damping = 0.0)
    dispatcher.add_gravity(gravity = GRAVITY)

    # Set render preferences
    orange_color = common.meta_data_render(1.0, 'orange', 'solid', 0.8)
    blue_color = common.meta_data_render(1.0, 'blue', 'solid')
    green_color = common.meta_data_render(1.0, 'green', 'solid')

    dispatcher.set_render_prefs(obj = 'wire', prefs = blue_color)
    dispatcher.set_render_prefs(obj = 'edge_constraint', prefs = green_color)
    dispatcher.set_render_prefs(obj = 'collider', prefs = orange_color)
    dispatcher.set_render_prefs(obj = 'anchor', prefs = orange_color)

    render.set_viewport_limit(-2.5, -2.5, 2.5, 2.5)
