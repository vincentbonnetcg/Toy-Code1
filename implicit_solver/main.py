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
NUM_SUBSTEP = 10 # number of substep per frame
RENDER_FOLDER_PATH = "" # specify a folder to export png files
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif

'''
 Run
'''
def main():
    # Create scene and solver
    scene = system.createBeamScene()
    solver = system.ImplicitSolver(FRAME_TIMESTEP / NUM_SUBSTEP, NUM_SUBSTEP)
    #solver = sl.SemiImplicitSolver(FRAME_TIMESTEP / NUM_SUBSTEP, NUM_SUBSTEP) #- only debugging - unstable with high stiffness
    
    # Run simulation and render
    render = tools.Render()
    render.setRenderFolderPath(RENDER_FOLDER_PATH)
    
    profiler = tools.profiler.ProfilerSingleton()
    for frameId in range(0, NUM_FRAME+1):
        profiler.clearLogs()
    
        if frameId > 0:
            solver.solveFrame(scene)
    
        render.showCurrentFrame(scene, frameId)
        render.exportCurrentFrame(str(frameId).zfill(4) + " .png")
    
        profiler.printLogs()
  
if __name__ == '__main__':
    main()