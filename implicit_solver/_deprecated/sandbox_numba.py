"""
@author: Vincent Bonnet
@description : Experimentations with Numba
"""

import numba
import numpy as np

def allocate_particles_as_sof(channel_names, num_particles, default_value):
    '''
    Allocate particle data as a structure of array (sof)
    '''
    array_dtype = []
    for name in channel_names:
        array_dtype.append((name, np.float32, (num_particles, 2)))

    particle_data = np.ones(1, dtype=np.dtype(array_dtype))[0]
    for name in channel_names:
        particle_data[name].fill(default_value)

    return particle_data


@numba.njit(debug=True)
def print_first_particle_system(object_ids, particle_data):
    for object_id in object_ids:
        print(particle_data[object_id])

#particle_data0 = allocate_particles_as_sof(['x', 'v', 'f'], 10, 0.0)
#particle_data1 = allocate_particles_as_sof(['x', 'v', 'f'], 10, 1.0)
#particle_data2 = allocate_particles_as_sof(['x', 'v', 'f', 'id'], 10, 2.0)
#objects = (particle_data0, particle_data1)
#print_first_particle_system([0, 1], objects)
#print_first_particle_system([0], (particle_data2,))


@numba.njit
def append_test(column, index, value, bla):
    
    result_list = []
    for i in range(bla):
        result_list.append({i:np.zeros((2,2))})
    
    #result_list[column][index]=value
    #result_list[column][index+1]=value

    return result_list

@numba.njit
def add(A, i, j, data):
    row = A[i]
    if j in row:
        row[j] += data
    else:
        row[j] = data

@numba.njit
def A(e):
    print(e)

@numba.njit
def B():
    e = [1.0, 2.0] 
    A(e)
    return e




#A = append_test(column =0, index=3, value=np.zeros((2,2)), bla=5)
#A = tuple(A)
#add(A, 0, 0, np.ones((2,2)))
#print(A[0][0])




