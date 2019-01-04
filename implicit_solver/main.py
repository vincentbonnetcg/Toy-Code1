"""
@author: Vincent Bonnet
@description : main
"""

import tools
import system

'''
 Global Constants
'''
NUM_FRAME = 100
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 4 # number of substep per frame
RENDER_FOLDER_PATH = "" # specify a folder to export png files
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif

def main():
    '''
    Creates a scene and a solver + solve
    '''
    # Create scene, solver and context
    scene = system.create_wire_scene()
    solver = system.ImplicitSolver()
    #solver = system.SemiImplicitSolver() # only for debugging - unstable
    context = system.Context(time = 0.0, dt = FRAME_TIMESTEP / NUM_SUBSTEP)
    solver.initialize(scene, context)

    # Creates render and profiler
    render = tools.Render()
    render.setRenderFolderPath(RENDER_FOLDER_PATH)
    profiler = tools.Profiler()

    # Simulate frames
    for frame_id in range(0, NUM_FRAME+1):
        profiler.clearLogs()

        if frame_id > 0:
            for _ in range(NUM_SUBSTEP):
                context.time += context.dt
                solver.solveStep(scene, context)

        render.showCurrentFrame(solver, scene, frame_id)
        render.exportCurrentFrame(str(frame_id).zfill(4) + " .png")

        profiler.printLogs()

if __name__ == '__main__':
    main()
