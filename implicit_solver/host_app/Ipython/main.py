"""
@author: Vincent Bonnet
@description : main
"""

import tools
import system
from render import Render
from host_app.dispatcher import CommandDispatcher
from host_app.client import Client
from tests import scene_examples

'''
 Global Constants
'''
START_TIME = 0
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 6 # number of substep per frame
NUM_FRAMES = 100 # number of simulated frame (doesn't include initial frame)
RENDER_FOLDER_PATH = "" # specify a folder to export png files
USE_REMOTE_SERVER = False # run the program locally or connect to a server
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif


def main():
    # Creates command dispatcher
    cmd_dispatcher= None
    if USE_REMOTE_SERVER:
        cmd_dispatcher = Client("Spyder")
        cmd_dispatcher.connect_to_server()
    else:
        cmd_dispatcher = CommandDispatcher()

    # Initialize dispatcher (context and scene)
    context = system.Context(time = START_TIME, frame_dt = FRAME_TIMESTEP,
                         num_substep = NUM_SUBSTEP, num_frames = NUM_FRAMES)

    cmd_dispatcher.run("set_context", context = context)
    scene_examples.init_wire_example(cmd_dispatcher)

    # Creates render and profiler
    render = Render()
    render.set_render_folder_path(RENDER_FOLDER_PATH)
    profiler = tools.Profiler()

    # Simulate frames
    for frame_id in range(context.num_frames+1):
        profiler.clearLogs()

        if frame_id == 0:
            cmd_dispatcher.run("initialize")
        else:
            cmd_dispatcher.run("solve_to_next_frame")

        render.show_current_frame(cmd_dispatcher, frame_id)
        render.export_current_frame(str(frame_id).zfill(4) + " .png")

        profiler.printLogs()

    # Disconnect client from server
    if USE_REMOTE_SERVER:
        cmd_dispatcher.disconnect_from_server()

if __name__ == '__main__':
    main()

