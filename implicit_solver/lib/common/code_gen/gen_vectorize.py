"""
@author: Vincent Bonnet
@description : Code Generation to convert function into numba friendly function
"""

# Package used for gen_vectorize
import inspect
import re
import functools

# Possible packages used by the generated functions
import numba
import numpy as np
import lib.common.node_accessor as na

def generate_vectorize_method(method, use_njit = True):
    '''
    Returns a tuple (source code, function object)
    '''
    generated_function_name = 'generated_' + method.__name__

    # Get code
    function_source = inspect.getsource(method)

    # Check arguments
    list_variable_names = []
    list_variable_accessors = []
    function_args = inspect.signature(method)
    parameter_remap = {} # dictionnary mapping object.attr with object_attr
    parameter_names = []
    for param in function_args.parameters:
        # regular expression to check whether the argument has a format object.attr
        param_attrs = re.findall(param+'[.][a-zA-Z0-9_]*', function_source)
        for param_attr in param_attrs:
            if param_attr in parameter_remap:
                continue
            # object.attr => object_attr (variable name)
            variable_name = param_attr.replace('.', '_')
            # object.attr into object['attr'] (variable accessor)
            object_attr = param_attr.split('.')
            variable_accessor = object_attr[0] +'[\'' + object_attr[1] + '\']'
            # add to list
            list_variable_names.append(variable_name)
            list_variable_accessors.append(variable_accessor)
            # store renaming
            parameter_remap[param_attr] = variable_name

        parameter_names.append(param)

    # Generate code
    code_lines = function_source.splitlines()
    gen_code_lines = []
    indents = ' ' * 4
    for code in code_lines:
        # remove any decorators from the function
        if code[0] == '@':
            continue

        if code[0:4] == 'def ':
            # add njit
            if use_njit:
                gen_code_lines.append('@numba.njit')
            # replace function name
            gen_code_lines.append('def '+generated_function_name+'('+ ', '.join(parameter_names) +'):')
            # add variable accessor
            num_variables = len(list_variable_names)
            for var_id in range(num_variables):
                variable_name = list_variable_names[var_id]
                variable_accessor = list_variable_accessors[var_id]
                variable_code = indents + variable_name + ' = ' + variable_accessor
                gen_code_lines.append(variable_code)
            # start loop on the first variable (should be a numpy array)
            variable_name = list_variable_names[0]
            gen_code_lines.append(indents + 'num_elements = ' + variable_name + '.shape[0]')
            gen_code_lines.append(indents + 'for i in range(num_elements):')
        else:
            for key, value in parameter_remap.items():
                code = code.replace(key, value+'[i]')

            gen_code_lines.append(indents + code)

    # Compile code
    generated_function_source = '\n'.join(gen_code_lines)
    generated_function_object = compile(generated_function_source, generated_function_name, 'exec')
    exec(generated_function_object)

    return generated_function_source, generated_function_name, vars().get(generated_function_name)

def as_vectorized(method, use_njit = True):
    '''
    Decorator from Datablock to Component
    '''
    @functools.wraps(method)
    def execute(*args):
        execute.generated_function(*args)
        return True

    source, name, function = generate_vectorize_method(method, use_njit)

    execute.generated_source = source
    execute.generated_function = function
    return execute
