"""
@author: Vincent Bonnet
@description : Code Generation to convert function into njit function
"""

# Package used by gen_vectorize.py
import functools
import lib.common as common
import lib.common.code_gen.code_gen_helper as gen

# Possible packages used by the generated functions
import numba
import numpy as np
import lib.common.node_accessor as na

def generate_njit_function(function):
    # Generate code
    helper = gen.CodeGenHelper(use_njit=True)
    helper.generate_njit_function_source(function)

    # Compile code
    generated_function_object = compile(helper.generated_function_source, helper.generated_function_name, 'exec')
    exec(generated_function_object)

    return helper.generated_function_source, vars().get(helper.generated_function_name)

def as_njit(function):
    '''
    Decorator to njit a function with Numba
    '''
    def convert(arg):
        '''
        Convert function argument into a type reconizable by numba
        Convert with the following rules
        List/Tuple => Tuple
        DataBlock => DataBlock.blocks
        Object => Object.DataBlock.blocks
        '''
        if isinstance(arg, (list, tuple)):
            new_arg = [None] * len(arg)
            for index, element in enumerate(arg):
                new_arg[index] = convert(element)
            return tuple(new_arg)
        elif isinstance(arg, common.DataBlock):
            return tuple(arg.blocks)
        elif hasattr(arg, 'data') and isinstance(arg.data, common.DataBlock):
            return tuple(arg.data.blocks)

        return arg

    @functools.wraps(function)
    def execute(*args):
        '''
        Execute the function. At least one argument is expected
        '''
        arg_list = list(args)

        # Fetch numpy array from common.DataBlock or a container of common.DataBlock
        for arg_id , arg in enumerate(arg_list):
            arg_list[arg_id] = convert(arg)

        # Call function
        return execute.generated_function(*arg_list)

    source, function = generate_njit_function(function)

    execute.generated_source = source
    execute.generated_function = function
    return execute