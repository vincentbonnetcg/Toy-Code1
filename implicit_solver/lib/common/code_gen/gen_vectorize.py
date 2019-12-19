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

def generate_vectorize_function(function, use_njit = True):
    '''
    Returns a tuple (source code, function object)
    '''
    # Generate code
    helper = gen.CodeGenHelper(use_njit)
    helper.generate_vectorized_function_source(function)

    # Compile code
    generated_function_object = compile(helper.generated_function_source, helper.generated_function_name, 'exec')
    exec(generated_function_object)

    return helper.generated_function_source, vars().get(helper.generated_function_name)

def as_vectorized(function, use_njit = True):
    '''
    Decorator to vectorize a function
    '''
    def convert(arg):
        '''
        Convert function argument into a type reconizable by numba
        Convert with the following rules
        List/Tuple => Tuple
        DataBlock => DataBlock.blocks
        Object => Object.DataBlock.blocks
        '''
        if isinstance(arg, common.DataBlock):
            if isinstance(arg.blocks, tuple):
                return arg.blocks
            else:
                raise ValueError("The blocks should be in a tuple. use datablock.lock()")

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
        if isinstance(args[0], (list, tuple)):
            new_arg_list = list(arg_list)
            for datablock in args[0]:
                if isinstance(datablock, common.DataBlock) and not datablock.isEmpty():
                    new_arg_list[0] = datablock
                    execute.generated_function(*new_arg_list)
                else:
                    raise ValueError("The first argument should be a datablock")
        elif isinstance(args[0], common.DataBlock):
                if not args[0].isEmpty():
                    execute.generated_function(*arg_list)
        else:
            raise ValueError("The first argument should be a datablock or a list of datablocks")

        return True

    source, function = generate_vectorize_function(function, use_njit)

    execute.generated_source = source
    execute.generated_function = function
    return execute
