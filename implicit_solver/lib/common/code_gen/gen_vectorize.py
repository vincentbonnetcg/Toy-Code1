"""
@author: Vincent Bonnet
@description : Code Generation to convert function into numba friendly function
"""

# Package used by gen_vectorize.py
import functools
import lib.common as common
import lib.common.code_gen.code_gen_helper as gen

# Possible packages used by the generated functions
# TODO : modify to get the import from original python file
import numba
import numpy as np
import lib.common.jit.node_accessor as na
import lib.common.jit.math_2d as math2D
import lib.objects.jit.utils.spring_lib as spring_lib
import lib.objects.jit.utils.area_lib as area_lib
import lib.objects.jit.utils.bending_lib as bending_lib
import lib.system.jit.sparse_matrix_lib as sparse_lib
from lib.objects import Kinematic

def generate_vectorize_function(function, options : gen.CodeGenOptions):
    '''
    Returns a tuple (source code, function object)
    '''
    # Generate code
    helper = gen.CodeGenHelper(options)
    helper.generate_vectorized_function_source(function)

    # Compile code
    generated_function_object = compile(helper.generated_function_source, '', 'exec')
    exec(generated_function_object)

    return helper.generated_function_source, locals().get(helper.generated_function_name)

def convert_argument(arg):
    '''
    From DataBlock to DataBlock.blocks
    '''
    if isinstance(arg, common.DataBlock):
        if isinstance(arg.blocks, numba.typed.List):
            return arg.blocks
        else:
            raise ValueError("The blocks should be in a tuple/numba.Typed.List. use datablock.lock()")

    return arg

def as_vectorized(function=None, local={} , **options):
    '''
    Decorator with arguments to vectorize a function
    '''
    gen_options = gen.CodeGenOptions(options)
    if function is None:
        return functools.partial(as_vectorized, **options)

    def isDatablock(value):
        '''
        Returns whether the argument 'arg' is a datablock
        a list/tuple of numpy.void (array of complex datatypes) is also consider as a datablock
        '''
        if isinstance(value, common.DataBlock):
            return True

        if isinstance(value,(list, tuple)):
            return isinstance(value[0], np.void)

        return False

    @functools.wraps(function)
    def execute(*args):
        '''
        Execute the function. At least one argument is expected
        From Book : Beazley, David, and Brian K. Jones. Python Cookbook: Recipes for Mastering Python 3. " O'Reilly Media, Inc.", 2013.
        In Section : 9.6. Defining a Decorator That Takes an Optional Argument
        '''
        # Fetch numpy array from common.DataBlock
        arg_list = list(args)
        for arg_id , arg in enumerate(arg_list):
            arg_list[arg_id] = convert_argument(arg)

        # Call function
        first_argument = args[0] # argument to vectorize
        if isDatablock(first_argument):
            if len(first_argument) > 0:
                execute.function(*arg_list)
        elif isinstance(first_argument, (list, tuple)):
            for datablock in first_argument:
                if isDatablock(datablock):
                    if len(datablock) > 0:
                        arg_list[0] = convert_argument(datablock)
                        execute.function(*arg_list)
                else:
                    raise ValueError("The first argument should be a datablock")
        else:
            raise ValueError("The first argument should be a datablock or a list of datablocks")

        return True

    source, function = generate_vectorize_function(function, gen_options)

    execute.options = gen_options
    execute.source = source
    execute.function = function

    return execute


