"""
@author: Vincent Bonnet
@description : main
"""

import render as rn
import core
import host_app.rpc as rpc
import lib.examples as scenes
from lib.dispatcher import CommandSolverDispatcher

'''
 Global Constants
'''
START_TIME = 0
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 12 # number of substep per frame
NUM_FRAMES = 100  # number of simulated frame (doesn't include initial frame)
RENDER_FOLDER_PATH = "" # specify a folder to export png files
USE_REMOTE_SERVER = False # run the program locally or connect to a server
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif

def get_command_dispatcher():
    if USE_REMOTE_SERVER:
        client = rpc.Client("Spyder")
        client.connect_to_server()
        cmd_dispatcher = client.get_dispatcher()
        return cmd_dispatcher

    cmd_dispatcher = CommandSolverDispatcher()
    return cmd_dispatcher

def main():
    # Creates render and profiler
    render = rn.Render()
    render.set_render_folder_path(RENDER_FOLDER_PATH)
    profiler = core.Profiler()

    # Creates command dispatcher (local or remote)
    cmd_dispatcher = get_command_dispatcher()

    # Initialize dispatcher (context and scene)
    cmd_dispatcher.set_context(time = START_TIME, frame_dt = FRAME_TIMESTEP,
                         num_substep = NUM_SUBSTEP, num_frames = NUM_FRAMES)

    scenes.rabbit.assemble(cmd_dispatcher, render)
    #scenes.cat.assemble(cmd_dispatcher, render)
    #scenes.multiwire.assemble(cmd_dispatcher, render)
    #scenes.beam.assemble(cmd_dispatcher, render)
    #scenes.wire.assemble(cmd_dispatcher, render)
    #scenes.rabbit_cat.assemble(cmd_dispatcher, render)

    # Simulate frames
    for frame_id in range(NUM_FRAMES+1):
        profiler.clear_logs()

        if frame_id == 0:
            cmd_dispatcher.initialize()
        else:
            cmd_dispatcher.solve_to_next_frame()

        render.show_current_frame(cmd_dispatcher, frame_id)
        render.export_current_frame(str(frame_id).zfill(4) + " .png")

        #profiler.print_logs()

if __name__ == '__main__':
    main()
