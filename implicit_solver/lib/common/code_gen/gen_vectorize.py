"""
@author: Vincent Bonnet
@description : Code Generation to convert function into numba friendly function
"""

# Package used by gen_vectorize.py
import functools
import lib.common as common
import lib.common.code_gen.code_gen_helper as gen

# Possible packages used by the generated functions
import numba
import numpy as np
import lib.common.node_accessor as na

def generate_guvectorize_function(function, use_njit = True):
    '''
    Returns at tuple (source code, function object)
    '''
    # TODO
    pass

def generate_vectorize_function(function, use_njit = True):
    '''
    Returns a tuple (source code, function object)
    '''
    # Generate code
    helper = gen.CodeGenHelper(use_njit)
    helper.generate_function_source(function)

    # Compile code
    generated_function_object = compile(helper.generated_function_source, helper.generated_function_name, 'exec')
    exec(generated_function_object)

    return helper.generated_function_source, vars().get(helper.generated_function_name)

def as_vectorized(function, use_njit = True):
    '''
    Decorator from Datablock to Component
    '''
    @functools.wraps(function)
    def execute(*args):
        arg_list = list(args)

        # Fetch numpy array from common.DataBlock or a container of common.DataBlock
        for arg_id , arg in enumerate(arg_list):
            if isinstance(arg, common.DataBlock):
                arg_list[arg_id] = arg.data
            elif hasattr(arg, 'data') and isinstance(arg.data, common.DataBlock):
                arg_list[arg_id] = arg.data.data

        # Call function
        execute.generated_function(*arg_list)

        return True

    source, function = generate_vectorize_function(function, use_njit)

    execute.generated_source = source
    execute.generated_function = function
    return execute
