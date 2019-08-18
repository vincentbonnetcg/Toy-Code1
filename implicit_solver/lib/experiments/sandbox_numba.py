"""
@author: Vincent Bonnet
@description : Experimentations with Numba
"""

import numba
import numpy as np

def allocate_particles_as_sof(num_particles):
    '''
    Allocate particle data as a structure of array (sof)
    '''
    array_dtype = []
    for name in ['x', 'v', 'f']:
        array_dtype.append((name, np.float32, (num_particles, 2)))

    particle_data = np.zeros(1, dtype=np.dtype(array_dtype))
    return particle_data

@numba.njit
def print_first_particle_system(array_data):
    indices = [0, 1, 2] # silly but trigger issue
    print(array_data[indices[2]][0]['x'])

# WORKS - only because array have same size
particle_data0 = allocate_particles_as_sof(10)
particle_data1 = allocate_particles_as_sof(10)
my_array = [particle_data0, particle_data1]
print_first_particle_system(my_array)

# DOESN'T WORK - array do not have same size
# ERROR 1 - can't unbox heterogeneous list
#particle_data0 = allocate_particles_as_sof(10)
#particle_data1 = allocate_particles_as_sof(15)
#my_array = [particle_data0, particle_data1]
#print_first_particle_system(my_array)

# DOESN'T WORK - array do not have same size
# ERROR 2 - Invalid use of Function(<built-in function getitem>)
#particle_data0 = allocate_particles_as_sof(10)
#particle_data1 = allocate_particles_as_sof(10)
#my_array = [particle_data0, particle_data1]
#print_first_particle_system(tuple(my_array))

