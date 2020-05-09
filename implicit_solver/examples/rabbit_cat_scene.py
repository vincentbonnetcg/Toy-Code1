"""
@author: Vincent Bonnet
@description : example scene with a rabbit and a cat !
"""
import os
from . import common
import host_app.rpc.shape_io as io_utils

NODE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2

def assemble(dispatcher, render):
    '''
    Initalizes a scene including a cat shape created by the Maya/mesh_converter.py
    Latest Maya/Houdini doesn't support Python 3.x hence cannot use client.py to send data
    '''
    dispatcher.run('reset')

    # Load Data from file
    filename = os.path.join(common.get_resources_folder(), 'rabbit.npz')
    rabbit_shape = io_utils.create_shape_from_npz_file(filename)

    filename = os.path.join(common.get_resources_folder(), 'cat.npz')
    cat_shape = io_utils.create_shape_from_npz_file(filename)
    cat_position, cat_rotation = cat_shape.compute_best_transform()
    cat_position[0] = -10
    cat_position[1] = -20
    cat_rotation = 30

    # Add objects to the solver
    collider_handle = dispatcher.run('add_kinematic', shape = cat_shape,
                                     position=cat_position, rotation=cat_rotation)


    mesh_handle = dispatcher.run('add_dynamic', shape = rabbit_shape, node_mass = NODE_MASS)

    edge_condition_handle = dispatcher.run('add_edge_constraint', dynamic = mesh_handle,
                                                           stiffness = 100.0, damping = 0.0)

    dispatcher.run('add_kinematic_collision', dynamic = mesh_handle, kinematic = collider_handle,
                                               stiffness = 50000.0, damping = 0.0)

    dispatcher.run('add_gravity', gravity = GRAVITY)

    # Set render preferences
    dispatcher.run('set_render_prefs', obj = mesh_handle,
                                       prefs = common.meta_data_render(1.0, 'grey', 'solid'))
    dispatcher.run('set_render_prefs', obj = edge_condition_handle,
                                       prefs = common.meta_data_render(1.0, 'blue', 'solid'))
    dispatcher.run('set_render_prefs', obj = collider_handle,
                                       prefs = common.meta_data_render(1.0, 'orange', 'solid', 0.8))


    render.set_viewport_limit(-20.0, -40.0, 20.0, 0.0)

