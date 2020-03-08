"""
@author: Vincent Bonnet
@description : Solver to orchestrate the step of a solver
"""

import lib.common as cm
import lib.objects.jit as cpn
from lib.system import Scene

class SolverContext:
    '''
    SolverContext to store time, time stepping, etc.
    '''
    def __init__(self, time = 0.0, frame_dt = 1.0/24.0, num_substep = 4, num_frames = 1):
        self.time = time # current time (in seconds)
        self.start_time = time # start time (in seconds)
        self.end_time = time + (num_frames * frame_dt) # end time (in seconds)
        self.frame_dt = frame_dt # time step on a single frame (in seconds)
        self.num_substep = num_substep # number of substep per frame
        self.dt = frame_dt / num_substep # simulation substep (in seconds)
        self.num_frames = num_frames # number of simulated frame (doesn't include initial frame)

class SolverDetails:
    '''
    List of datablocks
    '''
    def __init__(self):
        block_size = 100
        self.node = cm.DataBlock(cpn.Node, block_size) # dynamic objects
        self.area = cm.DataBlock(cpn.Area, block_size) # area constraints
        self.bending = cm.DataBlock(cpn.Bending, block_size) # bending constraints
        self.spring = cm.DataBlock(cpn.Spring, block_size) # spring constraints
        self.anchorSpring = cm.DataBlock(cpn.AnchorSpring, block_size) # anchor spring constraints

    def block_from_datatype(self, datatype):
        blocks = [self.node, self.area, self.bending, self.spring, self.anchorSpring]
        datatypes = [cpn.Node, cpn.Area, cpn.Bending, cpn.Spring, cpn.AnchorSpring]
        index = datatypes.index(datatype)
        return blocks[index]

    def dynamics(self):
        return [self.node]

    def conditions(self):
        return [self.area, self.bending, self.spring, self.anchorSpring]

class Solver:
    '''
    Base Solver
    '''
    def __init__(self, time_integrator = None):
        self.time_integrator = time_integrator

    def initialize(self, scene : Scene, details : SolverDetails, context : SolverContext):
        '''
        Initialize the scene
        '''
        scene.init_kinematics(context.start_time)
        scene.init_conditions(details)

    @cm.timeit
    def solve_step(self, scene : Scene, details : SolverDetails, context : SolverContext):
        '''
        Solve a single step (pre/step/post)
        '''
        self._pre_step(scene, details, context)
        self._step(scene, details, context)
        self._post_step(scene, details, context)

    @cm.timeit
    def _pre_step(self, scene : Scene, details : SolverDetails, context : SolverContext):
        scene.update_kinematics(context.time, context.dt)
        scene.update_conditions(details) # allocate dynamically new conditions

    @cm.timeit
    def _step(self, scene : Scene, details : SolverDetails, context : SolverContext):
        if self.time_integrator:
            self.time_integrator.prepare_system(scene, details, context.dt)
            self.time_integrator.assemble_system(details, context.dt)
            self.time_integrator.solve_system(details, context.dt)

    @cm.timeit
    def _post_step(self, scene, details, context):
        pass
