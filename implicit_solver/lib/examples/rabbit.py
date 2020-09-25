"""
@author: Vincent Bonnet
@description : example scene with a rabbit !
"""
import os
from . import common
import core.shape_io as io_utils
from core import RectangleShape

NODE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2

def assemble(dispatcher, render):
    '''
    Initalizes a scene including a cat shape created by the Maya/mesh_converter.py
    Latest Maya/Houdini doesn't support Python 3.x hence cannot use client.py to send data
    '''
    file_path = 'rabbit.npz'
    dispatcher.reset()

    # Load Data from file
    filename = os.path.join(common.get_resources_folder(),file_path)
    shape = io_utils.create_shape_from_npz_file(filename)

    # Create collider 0
    anchor0_shape = RectangleShape(min_x=-5.0, min_y=4.0, max_x=4.5, max_y=5.0)
    anchor0_shape.compute_best_transform()
    anchor0_shape.transform((-7,-13), 30)

    # Create collider 1
    anchor1_shape = RectangleShape(min_x=-5.0, min_y=4.0, max_x=5.0, max_y=5.0)
    anchor1_shape.compute_best_transform()
    anchor1_shape.transform((13,-20), -45)

    # Create collider 2
    anchor2_shape = RectangleShape(min_x=-5.0, min_y=4.0, max_x=5.0, max_y=5.0)
    anchor2_shape.compute_best_transform()
    anchor2_shape.transform((0,-35), -45)

    # Add objects to the solver
    dispatcher.add_kinematic(shape = anchor0_shape, name = 'collider0')
    dispatcher.add_kinematic(shape = anchor1_shape, name = 'collider1')
    dispatcher.add_kinematic(shape = anchor2_shape, name = 'collider2')
    dispatcher.add_dynamic(shape = shape, node_mass = NODE_MASS, name = 'rabbit')

    dispatcher.add_edge_constraint(dynamic = 'rabbit', stiffness = 100.0,
                                   damping = 0.0, name = 'rabbit_edge')

    dispatcher.add_kinematic_collision(stiffness = 10000.0, damping = 0.0)

    dispatcher.add_gravity(gravity = GRAVITY)

    # Set render preferences
    orange_color = common.meta_data_render(1.0, 'orange', 'solid', 0.8)
    blue_color = common.meta_data_render(1.0, 'blue', 'solid')
    grey_color = common.meta_data_render(1.0, 'grey', 'solid')

    dispatcher.set_render_prefs(obj = 'rabbit', prefs = grey_color)
    dispatcher.set_render_prefs(obj = 'rabbit_edge', prefs = blue_color)
    dispatcher.set_render_prefs(obj = 'collider0', prefs = orange_color)
    dispatcher.set_render_prefs(obj = 'collider1', prefs = orange_color)
    dispatcher.set_render_prefs(obj = 'collider2', prefs = orange_color)

    render.set_viewport_limit(-35.0, -55.0, 35.0, -5.0)

