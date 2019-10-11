"""
@author: Vincent Bonnet
@description : main
"""

import render as rn
import lib.common as common
import lib.system as system
import logic.scene_examples as scene_examples
import logic.commands_lib as sim_cmds
import host_app.rpc as rpc

'''
 Global Constants
'''
START_TIME = 0
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 10 # number of substep per frame
NUM_FRAMES = 100 # number of simulated frame (doesn't include initial frame)
RENDER_FOLDER_PATH = "" # specify a folder to export png files
USE_REMOTE_SERVER = False # run the program locally or connect to a server
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif

def get_command_dispatcher():
    if USE_REMOTE_SERVER:
        cmd_dispatcher = rpc.Client("Spyder")
        cmd_dispatcher.connect_to_server()
        return cmd_dispatcher

    cmd_dispatcher = rpc.CommandDispatcher()
    cmd_dispatcher.register_cmd(sim_cmds.initialize)
    cmd_dispatcher.register_cmd(sim_cmds.solve_to_next_frame)
    return cmd_dispatcher

def main():
    # Creates render and profiler
    render = rn.Render()
    render.set_render_folder_path(RENDER_FOLDER_PATH)
    profiler = common.Profiler()

    # Creates command dispatcher (local or remote)
    cmd_dispatcher= get_command_dispatcher()

    # Initialize dispatcher (context and scene)
    context = system.Context(time = START_TIME, frame_dt = FRAME_TIMESTEP,
                         num_substep = NUM_SUBSTEP, num_frames = NUM_FRAMES)

    cmd_dispatcher.run("set_context", context = context)
    #scene_examples.init_cat_scene(cmd_dispatcher, render)
    #scene_examples.init_beam_example(cmd_dispatcher, render)
    scene_examples.init_wire_example(cmd_dispatcher, render)

    # Simulate frames
    for frame_id in range(context.num_frames+1):
        profiler.clear_logs()

        if frame_id == 0:
            cmd_dispatcher.run("initialize")
        else:
            cmd_dispatcher.run("solve_to_next_frame")

        render.show_current_frame(cmd_dispatcher, frame_id)
        render.export_current_frame(str(frame_id).zfill(4) + " .png")


        profiler.print_logs()

    # Disconnect client from server
    if USE_REMOTE_SERVER:
        cmd_dispatcher.disconnect_from_server()

if __name__ == '__main__':
    main()
