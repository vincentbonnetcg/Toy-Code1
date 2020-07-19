"""
@author: Vincent Bonnet
@description : line search algorithms

Linesearche algorithms try to minimize h(a)=f(x+ap) by solving h'(a)=0 where
    - p search direction
    - f object function
    - step size
"""
import numpy as np

SCALE_STEP = 0.1

def scale(function, guess, gradient, search_dir):
    return SCALE_STEP

BACKTRACKING_MAX_STEP_SIZE = 1.0
BACKTRACING_CONTROL = 0.5
BACKTRACKING_SHRINK_FACTOR = 0.5
BACKTRACKING_MAX_ITERATION = 10

def backtracking(function, guess, gradient, search_dir):
    m = np.dot(gradient, search_dir)
    t = -BACKTRACING_CONTROL * m

    value = function.value(guess)
    terminate = False
    j = 0
    step_size = BACKTRACKING_MAX_STEP_SIZE
    while not terminate:
        # stopping criteria
        if j >= BACKTRACKING_MAX_ITERATION:
            terminate = True

        test = value - function.value(guess+search_dir*step_size)
        if test >= step_size*t:
            terminate = True

        step_size *= BACKTRACKING_SHRINK_FACTOR
        j += 1

    return step_size
