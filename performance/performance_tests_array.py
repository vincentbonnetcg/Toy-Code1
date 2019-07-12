"""
@author: Vincent Bonnet
@description : Evaluate CPU and stuff
"""

import time
import math
import sys
import numpy as np
from numba import njit
import numba
import matplotlib.pyplot as plt
import matplotlib as mpl

def create_data(N):
    '''
    Create random numbers
    '''
    return np.random.rand(N)

@njit
def function(value):
    '''
    Operation on a single value
    '''
    return math.sqrt(math.tan(value) * value * math.cos(value))

'''
Algorithms
 - Pure Python Loop
 - Numpy Vectorized
 - Numba Single Thread
'''
def python_loop(array):
    for i in range(array.shape[0]):
        array[i] = function(array[i])
    return array

vectorized_function = np.vectorize(function)

@njit
def numba_loop(array):
    for i in numba.prange(array.shape[0]):
        array[i] = function(array[i])
    return array

@njit(parallel=True)
def numba_loop_threaded(array):
    for i in numba.prange(array.shape[0]):
        array[i] = function(array[i])
    return array

'''
Test Algorithms
'''
print("Python Distribution")
print(sys.version)

# Prepare plot
font = {'color':  'darkblue',
         'weight': 'normal',
         'size': 18}
mpl.style.use("seaborn")
plt.title('Python performance tests - array', fontdict=font)
plt.xlabel('array size')
plt.ylabel('time (s)')

plot_colours = ["xkcd:aqua",
            "xkcd:plum",
            "xkcd:teal",
            "xkcd:chartreuse",
            "xkcd:olive",
            "xkcd:green",
            "xkcd:red"]

# Run Tests
array_sizes = [1e4, 1e5, 1e6, 1e7]
algorithms = [python_loop, vectorized_function, numba_loop, numba_loop_threaded]
algorithm_names = ['Pure Python', 'Numpy.Vectorized', 'Numba Single Thread', 'Numba Threaded' ]

for algo_id in range(len(algorithms)):

    algorithm = algorithms[algo_id]
    algorithm_name = algorithm_names[algo_id]

    time_values = np.zeros(len(array_sizes))

    for size_id in range(len(array_sizes)):

        array = create_data(int(array_sizes[size_id]))

        # the line below is only to compile function in jit mode
        dummy_array = np.random.rand(1)
        algorithm(dummy_array)

        start_time = time.time()
        algorithm(array)
        end_time = time.time()
        computationTime = end_time - start_time
        time_values[size_id] = computationTime
        
        log = algorithm_name + ' %f sec' % (computationTime)
        print(log)

    print(array_sizes)
    plt.plot(array_sizes, time_values, color=plot_colours[algo_id], label=algorithm_name)

# Complete plot
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.show()
