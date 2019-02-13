"""
@author: Vincent Bonnet
@description : commands to run simulation
"""


def solve_to_next_frame(scene, solver, context):
    '''
    Solve the scene and move to the next time
    '''
    for _ in range(context.num_substep):
        context.time += context.dt
        solver.solveStep(scene, context)

def initialize(scene, solver, context):
    '''
    Initialize the solver
    '''
    solver.initialize(scene, context)
