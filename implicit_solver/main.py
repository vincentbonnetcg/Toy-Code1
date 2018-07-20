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
ROD_ROOT_POS = (0., 0.) # in meters
ROD_NUM_SEGMENTS = 4
ROD_LENGTH = 2.0 # in meters
ROD_STIFFNESS = 1.0 # in newtons per meter (N/m)
ROD_DAMPING = 0.01
ROD_PARTICLE_MASS = 0.001 # in Kg
GRAVITY = (0.0, -9.81) # in meters per second^2
NUM_FRAME = 20;
FRAME_TIMESTEP = 1.0/24.0 # in seconds
NUM_SUBSTEP = 4 # number of substep per frame

'''
 Execute
'''
wire = obj.Wire(ROD_ROOT_POS, ROD_LENGTH, ROD_NUM_SEGMENTS, ROD_PARTICLE_MASS, ROD_STIFFNESS, ROD_DAMPING)
ds.draw(wire, 0)

for frameId in range(1, NUM_FRAME+1): 
    print("")
    dt = FRAME_TIMESTEP / NUM_SUBSTEP
    for substepId in range(NUM_SUBSTEP): 
        sl.semiImplicitStep(wire, dt, GRAVITY)
        #sl.implicitStep(wire, dt, GRAVITY)
    ds.draw(wire, frameId)