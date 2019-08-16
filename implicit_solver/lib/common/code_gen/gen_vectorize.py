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

    def convert(arg):
        '''
        Convert function argument into a type reconizable by numba
        Convert with the following rules
        List/Tuple => Tuple
        DataBlock => DataBlock.data
        Object => Object.DataBlock.data
        '''
        if isinstance(arg, (list, tuple)):
            new_arg = [None] * len(arg)
            for index, element in enumerate(arg):
                new_arg[index] = convert(element)
            return tuple(new_arg)
        elif isinstance(arg, common.DataBlock):
            return arg.data
        elif hasattr(arg, 'data') and isinstance(arg.data, common.DataBlock):
            return arg.data.data

        return arg

    @functools.wraps(function)
    def execute(*args):
        arg_list = list(args)

        # Fetch numpy array from common.DataBlock or a container of common.DataBlock
        for arg_id , arg in enumerate(arg_list):
            arg_list[arg_id] = convert(arg)

        # Call function
        if len(arg_list) > 0 and isinstance(arg_list[0], (list, tuple)):
            new_arg_list = list(arg_list)
            for element in new_arg_list[0]:
                new_arg_list[0] = element
                execute.generated_function(*new_arg_list)
        else:
            execute.generated_function(*arg_list)

        return True

    source, function = generate_vectorize_function(function, use_njit)

    execute.generated_source = source
    execute.generated_function = function
    return execute
