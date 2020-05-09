"""
@author: Vincent Bonnet
@description : example scene with a rabbit !
"""
import logic
from . import common
import host_app.rpc.shape_io as io_utils

NODE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2

def assemble(dispatcher, render):
    '''
    Initalizes a scene including a cat shape created by the Maya/mesh_converter.py
    Latest Maya/Houdini doesn't support Python 3.x hence cannot use client.py to send data
    '''
    file_path = 'rabbit.npz'
    dispatcher.run('reset')

    # Load Data from file
    filename = common.get_resources_folder() + file_path
    shape = io_utils.create_shape_from_npz_file(filename)

    # Create collider 0
    anchor0_shape = logic.RectangleShape(min_x=-5.0, min_y=4.0, max_x=4.5, max_y=5.0)
    anchor0_position, anchor_rotation = anchor0_shape.compute_best_transform()
    anchor0_position[0] = -7
    anchor0_position[1] = -13
    anchor0_rotation = 30

    # Create collider 1
    anchor1_shape = logic.RectangleShape(min_x=-5.0, min_y=4.0, max_x=5.0, max_y=5.0)
    anchor1_position, anchor_rotation = anchor1_shape.compute_best_transform()
    anchor1_position[0] = 13
    anchor1_position[1] = -20
    anchor1_rotation = -45

    # Create collider 2
    anchor2_shape = logic.RectangleShape(min_x=-5.0, min_y=4.0, max_x=5.0, max_y=5.0)
    anchor2_position, anchor_rotation = anchor2_shape.compute_best_transform()
    anchor2_position[0] = 0
    anchor2_position[1] = -30
    anchor2_rotation = 45

    # Add objects to the solver
    collider0_handle = dispatcher.run('add_kinematic', shape = anchor0_shape,
                                                         position = anchor0_position,
                                                         rotation = anchor0_rotation)

    collider1_handle = dispatcher.run('add_kinematic', shape = anchor1_shape,
                                                         position = anchor1_position,
                                                         rotation = anchor1_rotation)

    collider2_handle = dispatcher.run('add_kinematic', shape = anchor2_shape,
                                                         position = anchor2_position,
                                                         rotation = anchor2_rotation)

    mesh_handle = dispatcher.run('add_dynamic', shape = shape, node_mass = NODE_MASS)

    edge_condition_handle = dispatcher.run('add_edge_constraint', dynamic = mesh_handle,
                                                           stiffness = 100.0, damping = 0.0)

    dispatcher.run('add_kinematic_collision', dynamic = mesh_handle, kinematic = collider0_handle,
                                               stiffness = 10000.0, damping = 0.0)

    dispatcher.run('add_kinematic_collision', dynamic = mesh_handle, kinematic = collider1_handle,
                                               stiffness = 10000.0, damping = 0.0)

    dispatcher.run('add_kinematic_collision', dynamic = mesh_handle, kinematic = collider2_handle,
                                               stiffness = 10000.0, damping = 0.0)

    dispatcher.run('add_gravity', gravity = GRAVITY)

    # Set render preferences
    dispatcher.run('set_render_prefs', obj = mesh_handle,
                                       prefs = common.meta_data_render(1.0, 'grey', 'solid'))
    dispatcher.run('set_render_prefs', obj = edge_condition_handle,
                                       prefs = common.meta_data_render(1.0, 'blue', 'solid'))
    dispatcher.run('set_render_prefs', obj = collider0_handle,
                                       prefs = common.meta_data_render(1.0, 'orange', 'solid', 0.8))
    dispatcher.run('set_render_prefs', obj = collider1_handle,
                                       prefs = common.meta_data_render(1.0, 'orange', 'solid', 0.8))
    dispatcher.run('set_render_prefs', obj = collider2_handle,
                                       prefs = common.meta_data_render(1.0, 'orange', 'solid', 0.8))


    render.set_viewport_limit(-20.0, -40.0, 20.0, 0.0)

