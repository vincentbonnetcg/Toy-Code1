"""
@author: Vincent Bonnet
@description : Solver to orchestrate the step of a solver
"""

import core
from lib.system import Scene
from core import Details

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


class Solver:
    '''
    Solver Implementation
    '''
    def __init__(self, time_integrator):
        self.time_integrator = time_integrator

    def initialize(self, scene : Scene, details : Details, context : SolverContext):
        '''
        Initialize the scene
        '''
        scene.init_kinematics(details, context)
        scene.init_conditions(details)

    @core.timeit
    def solve_step(self, scene : Scene, details : Details, context : SolverContext):
        '''
        Solve a single step (pre/step/post)
        '''
        self._pre_step(scene, details, context)
        self._step(scene, details, context)
        self._post_step(details, context)

    @core.timeit
    def _pre_step(self, scene : Scene, details : Details, context : SolverContext):
        scene.update_kinematics(details, context)
        scene.update_conditions(details) # allocate dynamically new conditions

    @core.timeit
    def _step(self, scene : Scene, details : Details, context : SolverContext):
        self.time_integrator.prepare_system(scene, details, context.dt)
        self.time_integrator.assemble_system(details, context.dt)
        self.time_integrator.solve_system(details, context.dt)

    @core.timeit
    def _post_step(self, details, context):
        pass
