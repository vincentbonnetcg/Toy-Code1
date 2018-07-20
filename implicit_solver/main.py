"""
@author: Vincent Bonnet
@description : main
"""

import objects as obj
import display as ds
import solvers as sl

'''
 Global Constants
'''
WIRE_ROOT_POS = [0., 0.] # in meters
WIRE_LENGTH = 2.0 # in meters
WIRE_NUM_SEGMENTS = 4

BEAM_POS = [-4.0, 0.0] # in meters
BEAM_WIDTH = 8.0 # in meters
BEAM_HEIGHT = 1.0 # in meters
BEAM_CELL_X = 6 # number of cells along x
BEAM_CELL_Y = 3 # number of cells along y

STIFFNESS = 1.0 # in newtons per meter (N/m)
DAMPING = 0.01
PARTICLE_MASS = 0.001 # in Kg

GRAVITY = (0.0, -9.81) # in meters per second^2
NUM_FRAME = 100;
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 4 # number of substep per frame

'''
 Execute
'''
wire = obj.Wire(WIRE_ROOT_POS, WIRE_LENGTH, WIRE_NUM_SEGMENTS, PARTICLE_MASS, STIFFNESS, DAMPING)
beam = obj.Beam(BEAM_POS, BEAM_WIDTH, BEAM_HEIGHT, BEAM_CELL_X, BEAM_CELL_Y, PARTICLE_MASS, STIFFNESS, DAMPING)

simulatedObj = beam
ds.draw(simulatedObj, 0)

for frameId in range(1, NUM_FRAME+1): 
    print("")
    dt = FRAME_TIMESTEP / NUM_SUBSTEP
    for substepId in range(NUM_SUBSTEP): 
        sl.semiImplicitStep(simulatedObj, dt, GRAVITY)
        #sl.implicitStep(simulatedObj, dt, GRAVITY)
    ds.draw(simulatedObj, frameId)
    
