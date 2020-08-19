"""
@author: Vincent Bonnet
@description : Solver to orchestrate the step of a solver
"""

import lib.common as cm
from lib.objects.jit.data import Node, Area, Bending, Spring, AnchorSpring
from lib.objects.jit.data import Point, Edge, Triangle, Tetrahedron
from lib.system import Scene
from collections import namedtuple

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
    Details contains the datablocks
    '''
    def __init__(self):
        system_types = [Node, Area, Bending, Spring, AnchorSpring]
        system_types += [Point, Edge, Triangle, Tetrahedron]

        self.attribute_names = []
        self.datablocks = []

        # create datablock
        block_size = 100
        for datatype in system_types:
            datablock = cm.DataBlock(datatype, block_size)
            setattr(self, datatype.name(), datablock) # TODO : will be removed in the future !
            self.attribute_names.append(datatype.name())
            self.datablocks.append(datablock)

        # create bundle (namedtuple) accessible by numba
        blocks = [data.blocks for data in self.datablocks]
        self.bundleType = namedtuple('solveDetails', self.attribute_names)
        self.bundle = self.bundleType(*blocks)

        # create groups (subset of bundle)
        # TODO - replace with a nametuple
        group = lambda types : [self.datablocks[system_types.index(datatype)].blocks
                                for datatype in types]
        self.dynamic_group = group([Node])
        self.condition_group = group([Area, Bending, Spring, AnchorSpring])

    def block_from_datatype(self, typename):
        index = self.attribute_names.index(typename)
        return self.datablocks[index]

    def dynamics(self):
        return self.dynamic_group

    def conditions(self):
        return self.condition_group

class Solver:
    '''
    Solver Implementation
    '''
    def __init__(self, time_integrator):
        self.time_integrator = time_integrator

    def initialize(self, scene : Scene, details : SolverDetails, context : SolverContext):
        '''
        Initialize the scene
        '''
        scene.init_kinematics(details, context.start_time)
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
        scene.update_kinematics(details, context.time, context.dt)
        scene.update_conditions(details) # allocate dynamically new conditions

    @cm.timeit
    def _step(self, scene : Scene, details : SolverDetails, context : SolverContext):
        self.time_integrator.prepare_system(scene, details, context.dt)
        self.time_integrator.assemble_system(details, context.dt)
        self.time_integrator.solve_system(details, context.dt)

    @cm.timeit
    def _post_step(self, scene, details, context):
        pass
