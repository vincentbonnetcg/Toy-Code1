"""
@author: Vincent Bonnet
@description : main
"""

import tools
import system
import render as rn
import host_app.ipc as ipc
import system.commands as sim_cmds
import system.setup.commands as setup_cmds

'''
 Global Constants
'''
START_TIME = 0
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 4 # number of substep per frame
NUM_FRAMES = 100 # number of simulated frame (doesn't include initial frame)
RENDER_FOLDER_PATH = "" # specify a folder to export png files
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif

def main():
    # Creates scene, solver and context
    scene = system.Scene()
    solver = system.ImplicitSolver()
    context = system.Context(time = START_TIME, frame_dt = FRAME_TIMESTEP,
                             num_substep = NUM_SUBSTEP, num_frames = NUM_FRAMES)
    system.init_wire_scene(scene, context)

    # Creates client and connect to server
    client = ipc.Client(scene, solver, context)
    # client.create_scene(..)
    # client.create_solver(..)
    # client.create_context(..)
    #client.connect_to_external_server(host = "localhost", port = 8080)

    # Creates render and profiler
    render = rn.Render()
    render.setRenderFolderPath(RENDER_FOLDER_PATH)
    profiler = tools.Profiler()

    # Simulate frames
    scene = client.scene()
    solver = client.solver()
    context = client.context()

    for frame_id in range(context.num_frames+1):
        profiler.clearLogs()

        if frame_id == 0:
            sim_cmds.initialize(scene, solver, context)
        else:
            sim_cmds.solve_to_next_frame(scene, solver, context)

        render.showCurrentFrame(solver, scene, frame_id)
        render.exportCurrentFrame(str(frame_id).zfill(4) + " .png")

        profiler.printLogs()

    # Disconnect client from server
    client.disconnect_from_external_server()

if __name__ == '__main__':
    main()
