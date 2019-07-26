"""
@author: Vincent Bonnet
@description : Evaluation of Abstract Syntax Trees
"""

import inspect
import lib.common as common
import numpy as np
from numba_helper import numba_friendly
from numba import njit, vectorize
import re

'''
Datablock Functions
'''
class Vertex:
    def __init__(self):
        self.x = 1.2
        self.y = 1.5

def create_datablock():
    datablock = common.DataBlock()
    datablock.add_field_from_class(Vertex)
    datablock.initialize(10)
    return datablock

'''
Functions
'''
def test_vectorize(method):
    '''
    Decorator from Datablock to Component
    '''
    def execute(*args):
        generated_function_name = method.__name__ + "_generated"
        generated_function_ptr = vars().get(generated_function_name)
        if generated_function_ptr is None:
            # get code
            function_code = inspect.getsource(method)

            # check arguments
            function_args = inspect.signature(method)
            for param in function_args.parameters:
                # regular expression to check whether the argument is an object with attribute
                param_attrs = re.findall(param+'[.][a-zA-Z0-9_]*', function_code)
                print(param_attrs)

            # remove any decorators from the function
            code_lines = function_code.splitlines()
            new_code_lines = []
            for code in code_lines:
                if code[0] != '@':
                    new_code_lines.append(code)
            function_code = '\n'.join(new_code_lines)
            print(function_code)

        # TODO - call the generated function
        #generated_function_ptr(*arg_list)

        return True

    return execute


@test_vectorize
def test_add(v0 : Vertex, v1 : Vertex, other_value):
    v0.x += v1.xy5_8 + other_value
    v0.y += v1.y + other_value

@numba_friendly
def test_add_generated(v0, v1, other_value):
    '''
    array_v0 contains {x, y}
    '''
    v0_x = v0['x']
    v0_y = v0['y']
    v1_x = v1['x']
    v1_y = v1['y']

    for i in range(v0_x.shape[0]):
        v0_x[i] += v1_x[i]
        v0_y[i] += v1_y[i]

'''
Test
'''
# TODO : generate test_add_generated(...) from test_add(...)
datablock0 = create_datablock()
datablock1 = create_datablock()
test_add(datablock0.data, datablock1.data)
