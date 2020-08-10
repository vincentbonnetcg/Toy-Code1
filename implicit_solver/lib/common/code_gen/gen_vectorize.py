"""
@author: Vincent Bonnet
@description : Code Generation to convert function into numba friendly function
"""

# Package used by gen_vectorize.py
import inspect
import functools
import numpy
import numba

import lib.common as common
import lib.common.code_gen.code_gen_helper as gen

def generate_vectorize_function(function, options : gen.CodeGenOptions):
    '''
    Returns a tuple (source code, function object)
    '''
    func_module = inspect.getmodule(function)
    # Generate code
    helper = gen.CodeGenHelper(options)
    helper.generate_vectorized_function_source(function)

    # Compile code
    generated_function_object = compile(helper.generated_function_source, '', 'exec')
    exec(generated_function_object, func_module.__dict__)

    return helper.generated_function_source, getattr(func_module, helper.generated_function_name)

def convert_argument(arg):
    '''
    From DataBlock to DataBlock.blocks
    '''
    if isinstance(arg, common.DataBlock):
        if isinstance(arg.blocks, numba.typed.List):
            return arg.blocks
        else:
            raise ValueError("The blocks should be in a numba.Typed.List.")

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
            return isinstance(value[0], numpy.void)

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
                if not isDatablock(datablock):
                    raise ValueError("The first argument should be a datablock")

                if len(datablock) > 0:
                    arg_list[0] = convert_argument(datablock)
                    execute.function(*arg_list)
        else:
            raise ValueError("The first argument should be a datablock or a list of datablocks")

        return True

    source, function = generate_vectorize_function(function, gen_options)

    execute.options = gen_options
    execute.source = source
    execute.function = function

    return execute


