"""
@author: Vincent Bonnet
@description : Code Generation to convert function into numba friendly function
"""

import inspect
import re
import numba

def generate_vectorize_method(method):

    generated_function_name = 'generated_' + method.__name__

    function_object = None
    does_function_exist = False

    if does_function_exist:
        # TODO
        pass
    else:
        # Get code
        function_code = inspect.getsource(method)

        # Check arguments
        list_variable_names = []
        list_variable_accessors = []
        function_args = inspect.signature(method)
        parameter_remap = {} # dictionnary mapping object.attr with object_attr
        for param in function_args.parameters:
            # regular expression to check whether the argument has a format object.attr
            param_attrs = re.findall(param+'[.][a-zA-Z0-9_]*', function_code)
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

        # Generate code
        code_lines = function_code.splitlines()
        gen_code_lines = []
        indents = ' ' * 4
        for code in code_lines:
            # remove any decorators from the function
            if code[0] == '@':
                continue

            if code[0:4] == 'def ':
                # add njit
                gen_code_lines.append('@numba.njit')
                # replace function name
                # TODO : remove indication
                gen_code_lines.append(code.replace(method.__name__, generated_function_name))
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
        function_code = '\n'.join(gen_code_lines)
        function_object = compile(function_code, generated_function_name, 'exec')
        exec(function_object)

    generated_function = vars().get(generated_function_name)
    return generated_function

def as_vectorized(method):
    '''
    Decorator from Datablock to Component
    '''
    def execute(*args):

        generated_function = generate_vectorize_method(method)
        print(generated_function)
        generated_function(*args)

        return True

    return execute
