"""
@author: Vincent Bonnet
@description : Code Generation Helper
"""
import inspect
import re

class CodeGenOptions:
    def __init__(self, options):
        self.njit = options.get('njit', True)
        self.parallel = options.get('parallel', False) # Unused
        self.debug = options.get('debug', False)
        self.fastmath = options.get('fastmath', False)
        self.block = options.get('block', False)

    def __str__(self):
        result = (
            f'njit {self.njit}, '
            f'parallel {self.parallel}, '
            f'debug {self.debug}, '
            f'fastmath {self.fastmath}, '
            f'block {self.block}')

        return result

class CodeGenWriter:
    def __init__(self, indent_size = 4):
        self.code_lines = []
        self.indent = 0
        self._indent_size = indent_size # space per indentation

    def append(self, code_line):
        if not code_line:
            self.code_lines.append(code_line)
            return

        indent_str = ' ' * self.indent * self._indent_size
        self.code_lines.append(indent_str + code_line)

    def add_lines(self, code_lines, obj_attrs_map, lambda_var_transform):
        # generate a dictionarry to update the code
        code_update = {}
        for obj, attrs in obj_attrs_map.items():
            for attr in attrs:
                code_update[obj+'.'+attr] = lambda_var_transform(obj, attr)

        # generate code
        for code in code_lines:
            for original_code, updated_code in code_update.items():
                code = code.replace(original_code, updated_code)
            self.append(code)

    def source(self):
        return '\n'.join(self.code_lines)

class CodeGenHelper:

    def __init__(self, options : CodeGenOptions):
        # Generated function
        self.generated_function_name = ''
        self.generated_function_source = ''
        # Arguments
        self.obj_attrs_map = {} # dictionnary to map object with all attributes
        self.functions_args = [] # original function arguments
        self.functions_defaults = [] # original function defaults
        # Options
        self.options = options
        # Test whether or not it makes sense
        if (not options.njit) and (options.parallel or options.debug):
            raise ValueError("Cannot use the flags {parallel, debug} when njit=False ")

    def generate_vectorized_function_source(self, function):
        '''
        Generate the source code of the function as a vectorized function
        '''
        writer = CodeGenWriter()

        #------------ #
        # Preparation #
        # ----------- #
        self.generated_function_name = 'vectorized_' + function.__name__

        # get source code
        function_source = inspect.getsource(function)
        function_signature = inspect.signature(function)
        code_lines = function_source.splitlines()

        # check arguments
        self.__prepare_arguments(function_source, function_signature)

        # find the beginning of the function body : just after 'def'
        function_body_line = 0
        for line_id, code in enumerate(code_lines):
            if code[0:4] == 'def ':
                function_body_line = line_id + 1

        # create arguments for the main and kernel functions
        vec_functions_interface = self.functions_args.copy() # arguments + defaults
        inner_kernel_args = self.functions_args.copy() # inner function arguments
        inner_kernel_call_args = self.functions_args.copy() # inner function call
        first_argument = f'{self.functions_args[0]}_blocks'
        for argId, arg in enumerate(self.functions_args):
            if arg in self.obj_attrs_map:
                vec_functions_interface[argId] += '_blocks'
                inner_kernel_args[argId] += '_block'
                inner_kernel_call_args[argId] = f'{vec_functions_interface[argId]}[_handle]'

            if self.functions_defaults[argId]:
                vec_functions_interface[argId] += '='+self.functions_defaults[argId]

        vec_functions_interface.append('_block_handles=None')

        # --------------- #
        # Code Generation #
        # --------------- #

        # generate code for the numba decorator (njit)
        if self.options.njit:
            numba_arguments = ('parallel','fastmath', 'debug')
            numba_default_options = (False, False, False)
            codegen_options = (self.options.parallel,
                               self.options.fastmath,
                               self.options.debug)

            args = []
            for i in range(len(codegen_options)):
                if numba_default_options[i] != codegen_options[i]:
                    arg = numba_arguments[i]+'='+str(codegen_options[i])
                    args.append(arg)

            if len(args)>0:
                arg = ','.join(args)
                writer.append('@numba.njit('+arg+')')
            else:
                writer.append('@numba.njit')

        # generate code for function interface
        writer.append(f'def {self.generated_function_name}('+ (', '.join(vec_functions_interface)) + '):')
        writer.indent += 1

        # generate code for the kernel function
        writer.append('def kernel('+', '.join(inner_kernel_args) +'):')
        writer.indent += 1
        if self.options.block:
            # add the code
            transform_variable = lambda obj, attr: f'{obj}_block[0][\'{attr}\']'
            writer.add_lines(code_lines[function_body_line:],
                             self.obj_attrs_map,
                             transform_variable)
        else:
            # add variable to access data
            for obj, attrs in self.obj_attrs_map.items():
                for attr in attrs:
                    variable_name =  f'_{obj}_{attr}'
                    variable_accessor = f'{obj}_block[0][\'{attr}\']'
                    writer.append(f'{variable_name} = {variable_accessor}')

            writer.append(f'_num_elements = {inner_kernel_args[0]}[0][\'blockInfo_size\']')
            writer.append('for _i in range(_num_elements):')
            writer.indent += 1

            # add the code
            transform_variable = lambda obj, attr: f'_{obj}_{attr}[_i]'
            writer.add_lines(code_lines[function_body_line:],
                             self.obj_attrs_map,
                             transform_variable)

            writer.indent -= 1

        writer.indent -= 1
        writer.append('\n')

        # generate code to iterator over blocks and call the kernel
        writer.append('if _block_handles is None:' )
        writer.indent += 1
        writer.append(f'_num_blocks = len({first_argument})')
        writer.append('for _handle in range(_num_blocks):')
        writer.indent += 1
        writer.append(f'_active = {first_argument}[_handle][0][\'blockInfo_active\']' )
        writer.append('if _active:')
        writer.indent += 1
        writer.append('kernel('+', '.join(inner_kernel_call_args) +')')
        writer.indent -= 3
        writer.append('else:')
        writer.indent += 1
        writer.append('_num_blocks = len(_block_handles)' )
        writer.append('for _i in range(_num_blocks):')
        writer.indent += 1
        writer.append('_handle = _block_handles[_i]')
        writer.append(f'_active = {first_argument}[_handle][0][\'blockInfo_active\']' )
        writer.append('if _active:')
        writer.indent += 1
        writer.append('kernel('+', '.join(inner_kernel_call_args) +')')

        # generate the code
        self.generated_function_source = writer.source()

    def __prepare_arguments(self, function_source, function_signature):
        self.obj_attrs_map = {}
        self.functions_args = []
        self.functions_defaults = []
        recorded_params = []

        for param_id, param_name in enumerate(function_signature.parameters):
            param = function_signature.parameters[param_name]

            # add argument name and default
            arg_default = ''
            if param.default is not inspect._empty:
                arg_default = str(param.default)

            self.functions_args.append(param_name)
            self.functions_defaults.append(arg_default)

            # Two conditions
            # An argument is considered a datablock when associated to an annotation
            # Annotation (node:Node, constraint:Constraint ...)
            # and only available for the first argument
            if (param_id == 0) or (param.annotation is not inspect._empty):
                # regular expression to check whether the argument has a format object.attr
                param_attrs = re.findall(param_name+'[.][a-zA-Z0-9_]*', function_source)
                for param_attr in param_attrs:
                    if param_attr in recorded_params:
                        continue

                    recorded_params.append(param_attr)

                    obj, attr = param_attr.split('.')
                    attrs = self.obj_attrs_map.get(obj, [])
                    attrs.append(attr)
                    self.obj_attrs_map[obj] = attrs

        # the code generator does a simple search/replace on the source code
        # the attributes has to be sorted from longest to shortest
        # example : {'spring': ['f', 'fv']} => {'spring': [fv', 'f']}
        for obj, attrs in self.obj_attrs_map.items():
            attrs.sort(key=len, reverse=True)
