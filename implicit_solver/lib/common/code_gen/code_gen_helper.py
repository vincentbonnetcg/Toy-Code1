"""
@author: Vincent Bonnet
@description : Code Generation Helper
"""
import inspect
import re

class CodeGenOptions:
    def __init__(self, options):
        self.njit = options.get('njit', True)
        self.parallel = options.get('parallel', False)
        self.debug = options.get('debug', False)
        self.fastmath = options.get('fastmath', False)
        self.block_handles = options.get('block_handles', False)

    def __str__(self):
        result = 'njit ' + str(self.njit) + '\n'
        result += 'parallel ' + str(self.parallel) + '\n'
        result += 'debug ' + str(self.debug) + '\n'
        result += 'block_handles ' + str(self.block_handles)

        return result

class CodeGenHelper:

    def __init__(self, options : CodeGenOptions):
        # Generated function
        self.generated_function_name = ''
        self.generated_function_source = ''
        # Arguments
        self.obj_attrs_map = {} # dictionnary to map object with all attributes
        self.variable_remap = {} # dictionnary mapping 'object.attr' with 'object_attr'
        self.functions_args = [] # original functions arguments
        # Options
        self.options = options
        # Test whether or not it makes sense
        if (not options.njit) and (options.parallel or options.debug):
            raise ValueError("Cannot use the flags {parallel, debug} when njit=False ")

    def generate_vectorized_function_source(self, function):
        '''
        Generate the source code of the function as a vectorized function
        '''
        generated_function_name = 'generated_' + function.__name__

        # Get code
        function_source = inspect.getsource(function)
        function_signature = inspect.signature(function)

        # Check arguments
        self.__prepare_arguments(function_source, function_signature)

        # Generate source code
        code_lines = function_source.splitlines()
        gen_code_lines = []
        indents = ' ' * 4
        two_indents = ' ' * 8
        three_indents = ' ' * 12
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
                        gen_code_lines.append('@numba.njit('+arg+')')
                    else:
                        gen_code_lines.append('@numba.njit')

                # rename the function arguments associated with an object
                new_functions_args = self.functions_args.copy()
                for argId, arg in enumerate(new_functions_args):
                    if arg in self.obj_attrs_map:
                        new_functions_args[argId] += '_blocks'

                # replace function
                if self.options.block_handles:
                    gen_code_lines.append('def '+generated_function_name+'('+ ', '.join(new_functions_args) +', block_handles):')
                else:
                    gen_code_lines.append('def '+generated_function_name+'('+ ', '.join(new_functions_args) +'):')

                # loop over the blocks (list/tuple of numpy array)
                if self.options.block_handles:
                    gen_code_lines.append(indents + '_num_blocks = len(block_handles)' )
                else:
                    gen_code_lines.append(indents + '_num_blocks = len(' + new_functions_args[0]  + ')' )

                if self.options.parallel:
                    gen_code_lines.append(indents + 'for _j in numba.prange(_num_blocks):')
                else:
                    gen_code_lines.append(indents + 'for _j in range(_num_blocks):')

                if self.options.block_handles:
                    gen_code_lines.append(two_indents + '_handle = block_handles[_j]')
                else:
                    gen_code_lines.append(two_indents + '_handle = _j')

                # add variable to access block info
                master_argument = self.functions_args[0]
                master_variable_name = master_argument + '_blocks[_handle][\'blockInfo_numElements\']'
                gen_code_lines.append(two_indents + '_num_elements = ' + master_variable_name)
                master_variable_name = master_argument + '_blocks[_handle][\'blockInfo_active\']'
                gen_code_lines.append(two_indents + '_active = ' + master_variable_name)
                gen_code_lines.append(two_indents + 'if not _active:')
                gen_code_lines.append(three_indents + 'continue')

                # add variable to access block data
                for obj, attrs in self.obj_attrs_map.items():
                    for attr in attrs:
                        variable_name = '_' + obj + '_' + attr
                        variable_accessor = obj +'_blocks[_handle][\'' + attr + '\']'
                        variable_code = two_indents + variable_name + ' = ' + variable_accessor
                        gen_code_lines.append(variable_code)

                # loop over the elements (numpy array)
                gen_code_lines.append(two_indents + 'for _i in range(_num_elements):')

                # generate the variable remap
                self.variable_remap = {}
                for obj, attrs in self.obj_attrs_map.items():
                    for attr in attrs:
                        original_name = obj + '.' + attr
                        variable_name = '_' + obj + '_' + attr
                        self.variable_remap[original_name] = variable_name
            else:
                # prevent certain variables - cheap solution for now but could be improved
                if 'block_handles' in code:
                    raise ValueError("Cannot use the reserved 'block_handles' variables")

                for key, value in self.variable_remap.items():
                    code = code.replace(key, value+'[_i]')

                gen_code_lines.append(two_indents + code)

        # Set generated function name and source
        self.generated_function_name = generated_function_name
        self.generated_function_source = '\n'.join(gen_code_lines)

    def __prepare_arguments(self, function_source, function_signature):
        self.obj_attrs_map = {}
        self.functions_args = []
        recorded_params = []

        for param_name in function_signature.parameters:

            # A function argument is considered as a datablock
            # when it is associated to an annotation (such as cpn.Node, cpn.ConstraintBased ...)
            # it is not generic, but the code generation works with this assumption for now (december 2019)
            annotation_type = function_signature.parameters[param_name].annotation
            if annotation_type is not inspect._empty:
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

            self.functions_args.append(param_name)
