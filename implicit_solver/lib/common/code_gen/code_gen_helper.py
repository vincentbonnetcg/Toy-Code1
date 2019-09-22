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
        self.obj_attrs_map = {} # dictionnary to map object with all attributes
        self.variable_remap = {} # dictionnary mapping 'object.attr' with 'object_attr'
        self.functions_args = [] # original functions arguments
        # Options
        self.use_njit = use_njit

    def generate_njit_function_source(self, function):
        '''
        Generate the source code of the function as a njit
        '''
        generated_function_name = 'generated_' + function.__name__

        # Get code
        function_source = inspect.getsource(function)

        # Generate source code
        code_lines = function_source.splitlines()
        gen_code_lines = []
        for code in code_lines:

            # empty line
            if not code:
                gen_code_lines.append('')
                continue

            # remove decorators
            if code[0] == '@':
                continue

            if code[0:4] == 'def ':
                # add njit
                #gen_code_lines.append('@numba.njit')
                code = code.replace(function.__name__, generated_function_name)

            gen_code_lines.append(code)

        # Set generated function name and source
        self.generated_function_name = generated_function_name
        self.generated_function_source = '\n'.join(gen_code_lines)

    def generate_vectorized_function_source(self, function):
        '''
        Generate the source code of the function as a vectorized function
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
        two_indents = ' ' * 8
        for code in code_lines:

            # empty line
            if not code:
                gen_code_lines.append('')
                continue

            # remove decorators
            if code[0] == '@':
                continue

            if code[0:4] == 'def ':
                # add njit
                if self.use_njit:
                    gen_code_lines.append('@numba.njit')

                # rename the function arguments associated with an object
                new_functions_args = self.functions_args.copy()
                for argId, arg in enumerate(new_functions_args):
                    if arg in self.obj_attrs_map:
                        new_functions_args[argId] += '_blocks'

                # replace function name
                gen_code_lines.append('def '+generated_function_name+'('+ ', '.join(new_functions_args) +'):')

                # loop over the blocks (list/tuple of numpy array)
                gen_code_lines.append(indents + '_num_blocks = len(' + new_functions_args[0]  + ')' )
                gen_code_lines.append(indents + 'for _j in range(_num_blocks):')

                # add variable accessor
                for obj, attrs in self.obj_attrs_map.items():
                    for attr in attrs:
                        variable_name = obj + '_' + attr
                        variable_accessor = obj +'_blocks' + '[_j][\'' + attr + '\']'
                        variable_code = two_indents + variable_name + ' = ' + variable_accessor
                        gen_code_lines.append(variable_code)

                # loop over the elements (numpy array)
                master_argument = self.functions_args[0]
                master_attr = self.obj_attrs_map.get(master_argument, ['unknown'])[0]
                master_variable_name = master_argument + '_' + master_attr
                gen_code_lines.append(two_indents + '_num_elements = ' + master_variable_name + '.shape[0]')
                gen_code_lines.append(two_indents + 'for _i in range(_num_elements):')

                # generate the variable remap
                self.variable_remap = {}
                for obj, attrs in self.obj_attrs_map.items():
                    for attr in attrs:
                        original_name = obj + '.' + attr
                        variable_name = obj + '_' + attr
                        self.variable_remap[original_name] = variable_name
            else:
                for key, value in self.variable_remap.items():
                    code = code.replace(key, value+'[_i]')

                gen_code_lines.append(two_indents + code)

        # Set generated function name and source
        self.generated_function_name = generated_function_name
        self.generated_function_source = '\n'.join(gen_code_lines)

    def __prepare_arguments(self, function_source, function_args):
        self.obj_attrs_map = {}
        self.functions_args = []
        recorded_params = []
        for param in function_args.parameters:
            # regular expression to check whether the argument has a format object.attr
            param_attrs = re.findall(param+'[.][a-zA-Z0-9_]*', function_source)
            for param_attr in param_attrs:
                if param_attr in recorded_params:
                    continue
                recorded_params.append(param_attr)

                obj, attr = param_attr.split('.')
                attrs = self.obj_attrs_map.get(obj, [])
                attrs.append(attr)
                self.obj_attrs_map[obj] = attrs

            self.functions_args.append(param)
