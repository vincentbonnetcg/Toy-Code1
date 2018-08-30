"""
@author: Vincent Bonnet
@description : main
"""

import dynamics as dyn
import kinematics as kin
import render as rd
import solvers as sl
import scene as sc
import profiler
import math

'''
 Global Constants
'''
WIRE_ROOT_POS = [0., 0.] # in meters
WIRE_END_POS = [2., 0.] # in meters
WIRE_NUM_SEGMENTS = 10

BEAM_POS = [-4.0, 0.0] # in meters
BEAM_WIDTH = 8.0 # in meters
BEAM_HEIGHT = 1.0 # in meters
BEAM_CELL_X = 6 # number of cells along x
BEAM_CELL_Y = 3 # number of cells along y

STIFFNESS = 2.0 # in newtons per meter (N/m)
DAMPING = 0.0
PARTICLE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2
NUM_FRAME = 100;
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 4 # number of substep per frame

RENDER_FOLDER_PATH = "" # specify a folder to export png files
# Used command  "magick -loop 0 -delay 4 *.png out.gif"  to convert from png to animated gif

'''
 Execute
'''
def createWireScene():
    # Create dynamic objects / kinematic objects / scene
    wire = dyn.Wire(WIRE_ROOT_POS, WIRE_END_POS, WIRE_NUM_SEGMENTS, PARTICLE_MASS, STIFFNESS, DAMPING)
    point = kin.PointKinematic(WIRE_ROOT_POS)
    pointPos = point.position
    movePoint = lambda time : [[pointPos[0] + math.sin(2.0 * time) * 0.1, pointPos[1] + math.sin(time * 4.0)], 0.0]
    point.animationFunc = movePoint
    
    scene = sc.Scene(GRAVITY)
    scene.addDynamic(wire)
    scene.addKinematic(point)
    scene.updateKinematics(0.0) # set kinematic objects at start frame
    scene.attachToKinematic(wire, point, 100.0, 0.0, 0.1)
    return scene

def createBeamScene():
    # Create dynamic objects / kinematic objects / scene
    beam = dyn.Beam(BEAM_POS, BEAM_WIDTH, BEAM_HEIGHT, BEAM_CELL_X, BEAM_CELL_Y, PARTICLE_MASS, STIFFNESS, DAMPING)
    skinStartPos = [BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT]
    skinEndPos = [BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1] + BEAM_HEIGHT]
    skin = dyn.Wire(skinStartPos, skinEndPos, BEAM_CELL_X * 8, PARTICLE_MASS * 0.1, STIFFNESS, DAMPING)
    
    leftAnchor = kin.RectangleKinematic(BEAM_POS[0] - 0.5, BEAM_POS[1], BEAM_POS[0], BEAM_POS[1] + BEAM_HEIGHT)
    rightAnchor = kin.RectangleKinematic(BEAM_POS[0] + BEAM_WIDTH, BEAM_POS[1], BEAM_POS[0] + BEAM_WIDTH + 0.5, BEAM_POS[1] + BEAM_HEIGHT)
    
    LPos = leftAnchor.position
    moveLeftAnchor = lambda time : [[LPos[0] + math.sin(2.0 * time) * 0.1, LPos[1] + math.sin(time * 4.0)], 0.0]
    leftAnchor.animationFunc = moveLeftAnchor

    RPos = rightAnchor.position
    moveRightAnchor = lambda time : [[RPos[0] + math.sin(2.0 * time) * -0.1, RPos[1]], 0.0]
    rightAnchor.animationFunc = moveRightAnchor

    scene = sc.Scene(GRAVITY)
    scene.addDynamic(beam)
    scene.addDynamic(skin)
    scene.addKinematic(leftAnchor)
    scene.addKinematic(rightAnchor)
    scene.updateKinematics(0.0) # set kinematic objects at start frame
    scene.attachToKinematic(beam, rightAnchor, 100.0, 0.0, 0.1)
    scene.attachToKinematic(beam, leftAnchor, 100.0, 0.0, 0.1)
    scene.attachToDynamic(beam, skin, 100.0, 0.0, 0.001)
   
    return scene 

scene = createBeamScene()


# Create Solver
#solver = sl.SemiImplicitSolver(FRAME_TIMESTEP / NUM_SUBSTEP, NUM_SUBSTEP) #- only debugging - unstable with beam
solver = sl.ImplicitSolver(FRAME_TIMESTEP / NUM_SUBSTEP, NUM_SUBSTEP)

# Run simulation and render
render = rd.Render()
render.setRenderFolderPath(RENDER_FOLDER_PATH)

profiler = profiler.ProfilerSingleton()
for frameId in range(1, NUM_FRAME+1):
    profiler.clearLogs()
    
    solver.solveFrame(scene)

    print("")
    render.showCurrentFrame(scene, frameId)
    render.exportCurrentFrame(str(frameId).zfill(4) + " .png")
    
    profiler.printLogs()
