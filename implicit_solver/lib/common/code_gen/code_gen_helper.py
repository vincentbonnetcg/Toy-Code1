"""
@author: Vincent Bonnet
@description : Code Generation Helper
"""
import inspect
import re

class CodeGenHelper:

    def __init__(self, use_njit = True):
        # Generated function
        self.generated_function_name = ''
        self.generated_function_source = ''
        # Arguments
        self.list_variable_names = [] # ["object_attr", ...]
        self.list_variable_accessors = [] # ["object['attr']", ...]
        self.parameter_remap = {} # dictionnary mapping 'object.attr' with 'object_attr'
        self.parameter_names = [] # original parameters attributes
        # Options
        self.use_njit = use_njit

    def generate_function_source(self, function):
        '''
        Build function attibute
        '''
        generated_function_name = 'generated_' + function.__name__

        # Get code
        function_source = inspect.getsource(function)
        function_args = inspect.signature(function)

        # Check arguments
        self.__prepare_arguments(function_source, function_args)

        # Generate source code
        code_lines = function_source.splitlines()
        gen_code_lines = []
        indents = ' ' * 4
        for code in code_lines:
            # remove any decorators from the function
            if code[0] == '@':
                continue

            if code[0:4] == 'def ':
                # add njit
                if self.use_njit:
                    gen_code_lines.append('@numba.njit')
                # replace function name
                gen_code_lines.append('def '+generated_function_name+'('+ ', '.join(self.parameter_names) +'):')
                # add variable accessor
                num_variables = len(self.list_variable_names)
                for var_id in range(num_variables):
                    variable_name = self.list_variable_names[var_id]
                    variable_accessor = self.list_variable_accessors[var_id]
                    variable_code = indents + variable_name + ' = ' + variable_accessor
                    gen_code_lines.append(variable_code)
                # start loop on the first variable (should be a numpy array)
                variable_name = self.list_variable_names[0]
                gen_code_lines.append(indents + 'num_elements = ' + variable_name + '.shape[0]')
                gen_code_lines.append(indents + 'for i in range(num_elements):')
            else:
                for key, value in self.parameter_remap.items():
                    code = code.replace(key, value+'[i]')

                gen_code_lines.append(indents + code)

        # Set generated function name and source
        self.generated_function_name = 'generated_' + function.__name__
        self.generated_function_source = '\n'.join(gen_code_lines)

    def __prepare_arguments(self, function_source, function_args):
        self.list_variable_names = []
        self.list_variable_accessors = []
        self.parameter_remap = {}
        self.parameter_names = []
        for param in function_args.parameters:
            # regular expression to check whether the argument has a format object.attr
            param_attrs = re.findall(param+'[.][a-zA-Z0-9_]*', function_source)
            for param_attr in param_attrs:
                if param_attr in self.parameter_remap:
                    continue
                # object.attr => object_attr (variable name)
                variable_name = param_attr.replace('.', '_')
                # object.attr into object['attr'] (variable accessor)
                object_attr = param_attr.split('.')
                variable_accessor = object_attr[0] +'[\'' + object_attr[1] + '\']'
                # add to list
                self.list_variable_names.append(variable_name)
                self.list_variable_accessors.append(variable_accessor)
                # store renaming
                self.parameter_remap[param_attr] = variable_name

            self.parameter_names.append(param)

