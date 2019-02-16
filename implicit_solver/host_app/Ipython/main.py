"""
@author: Vincent Bonnet
@description : main
"""

import tools
import system
from render import Render
from host_app.dispatcher import CommandDispatcher
import system.setup.tests as tests

'''
 Global Constants
'''
START_TIME = 0
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 4 # number of substep per frame
NUM_FRAMES = 100 # number of simulated frame (doesn't include initial frame)
RENDER_FOLDER_PATH = "" # specify a folder to export png files
USE_REMOTE_SERVER = False # run the program locally or connect to a server
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif

def main():

    # Creates dispatcher
    cmd_dispatcher= None
    if USE_REMOTE_SERVER:
        # NOT IMPLEMENTED
        #cmd_dispatcher = ipc.Client()
        #cmd_dispatcher.connect_to_external_server(host = "localhost", port = 8080)
        pass
    else:
        context = system.Context(time = START_TIME, frame_dt = FRAME_TIMESTEP,
                             num_substep = NUM_SUBSTEP, num_frames = NUM_FRAMES)
        cmd_dispatcher = CommandDispatcher(context)

    # Init bundle with example
    tests.init_wire_example(cmd_dispatcher)

    # Creates render and profiler
    render = Render()
    render.setRenderFolderPath(RENDER_FOLDER_PATH)
    profiler = tools.Profiler()

    # Simulate frames
    for frame_id in range(context.num_frames+1):
        profiler.clearLogs()

        if frame_id == 0:
            cmd_dispatcher.run("initialize")
        else:
            cmd_dispatcher.run("solve_to_next_frame")

        render.showCurrentFrame(cmd_dispatcher.solver(), cmd_dispatcher.scene(), frame_id)
        render.exportCurrentFrame(str(frame_id).zfill(4) + " .png")

        profiler.printLogs()

    # Disconnect client from server
    if USE_REMOTE_SERVER:
        cmd_dispatcher.disconnect_from_external_server()

if __name__ == '__main__':
    main()
