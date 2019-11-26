"""
@author: Vincent Bonnet
@description : Demo file to init the solver
"""
import lib
import host_app.rpc as rpc
from host_app.Ipython.render import Render
from logic import scene_examples, RectangleShape

import time

'''
 Global Constants
'''
START_TIME = 0
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 10 # number of substep per frame
NUM_FRAMES = 100 # number of simulated frame (doesn't include initial frame)

g_render = Render()
g_frame_id = 0
context = lib.system.SolverContext(time = START_TIME, frame_dt = FRAME_TIMESTEP,
                     num_substep = NUM_SUBSTEP, num_frames = NUM_FRAMES)

def get_dispatcher(name):
    cmd_dispatcher = rpc.Client(name)
    if not cmd_dispatcher.connect_to_server(ip='127.0.0.1', port=8013):
        print('Cannot connect to server')
    return cmd_dispatcher

def stage_init_wire_scene(client_name):
    '''
    Initialize the remote solver with wire
    '''
    cmd_dispatcher = get_dispatcher(client_name)

    cmd_dispatcher.run('set_context', context = context)
    scene_examples.init_wire_example(cmd_dispatcher, g_render)

    cmd_dispatcher.run('initialize')
    g_render.show_current_frame(cmd_dispatcher, g_frame_id)
    print('')

def stage_simulate_frames(num_frames, client_name):
    global g_frame_id
    cmd_dispatcher = get_dispatcher(client_name)
    for i in range(num_frames):
        g_frame_id += 1
        cmd_dispatcher.run('solve_to_next_frame')
        g_render.show_current_frame(cmd_dispatcher, g_frame_id)
        print('')

def stage_add_collider(client_name):
    global g_frame_id
    cmd_dispatcher = get_dispatcher(client_name)

    # anchor shape and animation
    rectangle_shape = RectangleShape(min_x = 0.2, min_y = 0.3, max_x = 1.5, max_y = 0.7)
    rectangle_position, rectangle_rotation = rectangle_shape.extract_transform_from_shape()
    func = lambda time: [[rectangle_position[0],
                          rectangle_position[1]], -5.0]

    rectangle_animator = lib.objects.Animator(func, context)

    rectangle_handle = cmd_dispatcher.run('add_kinematic', shape = rectangle_shape,
                                                          position = rectangle_position,
                                                          rotation = rectangle_rotation,
                                                          animator = rectangle_animator)

    dynamic_handles = cmd_dispatcher.run('get_dynamic_handles')

    cmd_dispatcher.run('add_kinematic_collision', dynamic = dynamic_handles[0], kinematic = rectangle_handle,
                                               stiffness = 1000.0, damping = 0.0)


def stage_close(client_name):
    cmd_dispatcher = get_dispatcher(client_name)
    cmd_dispatcher.disconnect_from_server()


if __name__ == '__main__':
    time.sleep(3)
    stage_init_wire_scene('My3DApp')
    stage_simulate_frames(20, 'My3DAppTwo')
    stage_add_collider('My3DAppThree')
    time.sleep(3)
    stage_simulate_frames(150, 'My3DAppAgain')
    stage_close('Spyder')

