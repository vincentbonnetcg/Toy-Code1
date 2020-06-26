"""
@author: Vincent Bonnet
@description : line search algorithms

Linesearche algorithms try to minimize h(a)=f(x+ap) by solving h'(a)=0 where
    - p search direction
    - f object function
    - step size
"""

SCALE_STEP = 1.0
def scale(function, start_point, search_dir):
    return SCALE_STEP


SHRINK_FACTOR = 0.5 # how much to shrink per search
MAX_BACKTRACKING_ITERATION = 2
def backtracking(function, start_point, search_dir):
    optimal_length  = SCALE_STEP

    terminate = False
    while not terminate:
        # TODO - compute the optimal lenght
        terminate = True

    return optimal_length
