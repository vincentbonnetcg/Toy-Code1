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
import lib.objects.components_jit.utils.spring_lib as spring_lib
import lib.objects.components_jit.utils.area_lib as area_lib
import lib.objects.components_jit.utils.bending_lib as bending_lib
import lib.system.jit.sparse_matrix_lib as sparse_lib
import lib.common.convex_hull as ch

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

def as_vectorized(function=None, local={} , **options):
    '''
    Decorator with arguments to vectorize a function
    '''
    gen_options = gen.CodeGenOptions(options)
    if function is None:
        return functools.partial(as_vectorized, **options)

    def convert(arg):
        '''
        From DataBlock to DataBlock.blocks
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
        From Book : Beazley, David, and Brian K. Jones. Python Cookbook: Recipes for Mastering Python 3. " O'Reilly Media, Inc.", 2013.
        In Section : 9.6. Defining a Decorator That Takes an Optional Argument
        '''
        # Fetch numpy array from common.DataBlock
        arg_list = list(args)
        for arg_id , arg in enumerate(arg_list):
            arg_list[arg_id] = convert(arg)

        # Call function
        first_argument = args[0] # argument to vectorize
        if isinstance(first_argument, (list, tuple)):
            for datablock in first_argument:
                if isinstance(datablock, common.DataBlock):
                    if not datablock.is_empty():
                        arg_list[0] = convert(datablock)
                        execute.function(*arg_list)
                else:
                    raise ValueError("The first argument should be a datablock")
        elif isinstance(first_argument, common.DataBlock):
                if not first_argument.is_empty():
                    execute.function(*arg_list)
        else:
            raise ValueError("The first argument should be a datablock or a list of datablocks")

        return True

    source, function = generate_vectorize_function(function, gen_options)

    execute.options = gen_options
    execute.source = source
    execute.function = function

    return execute
