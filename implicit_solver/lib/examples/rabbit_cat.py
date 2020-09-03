"""
@author: Vincent Bonnet
@description : example scene with a rabbit and a cat !
"""
import os
from . import common
import core.shape_io as io_utils

NODE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2

def assemble(dispatcher, render):
    '''
    Initalizes a scene including a cat shape created by the Maya/mesh_converter.py
    Latest Maya/Houdini doesn't support Python 3.x hence cannot use client.py to send data
    '''
    dispatcher.reset()

    # Load Data from file
    filename = os.path.join(common.get_resources_folder(), 'rabbit.npz')
    rabbit_shape = io_utils.create_shape_from_npz_file(filename)

    filename = os.path.join(common.get_resources_folder(), 'cat.npz')
    cat_shape = io_utils.create_shape_from_npz_file(filename)
    cat_position, cat_rotation = cat_shape.compute_best_transform()
    cat_shape.transform((-10,-20), 30)

    # Add objects to the solver
    dispatcher.add_kinematic(shape = cat_shape, name = 'collider')
    dispatcher.add_dynamic(shape = rabbit_shape, node_mass = NODE_MASS, name = 'rabbit')

    dispatcher.add_edge_constraint(dynamic = 'rabbit', stiffness = 100.0, damping = 0.0, name='rabbit_edge')
    dispatcher.add_kinematic_collision(stiffness = 50000.0, damping = 0.0)

    dispatcher.add_gravity(gravity = GRAVITY)

    # Set render preferences
    orange_color = common.meta_data_render(1.0, 'orange', 'solid', 0.8)
    grey_color = common.meta_data_render(1.0, 'grey', 'solid')
    blue_color = common.meta_data_render(1.0, 'blue', 'solid')

    dispatcher.set_render_prefs(obj = 'rabbit', prefs = grey_color)
    dispatcher.set_render_prefs(obj = 'rabbit_edge', prefs = blue_color)
    dispatcher.set_render_prefs(obj = 'collider', prefs = orange_color)

    render.set_viewport_limit(-20.0, -40.0, 20.0, 0.0)

