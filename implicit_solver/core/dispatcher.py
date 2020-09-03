"""
@author: Vincent Bonnet
@description : Run command and bundle scene/solver/context together
"""

# import for CommandDispatcher
import functools
import inspect

class CommandDispatcher:
    '''
    Base class to developer command dispatcher
    '''
    def __init__(self):
        # list of registered commands
        self._commands = {}

    def register_cmd(self, cmd, cmd_name = None):
        if cmd_name is None:
            cmd_name = cmd.__name__

        if hasattr(self, cmd_name):
            raise ValueError(f'in register_cmd() {cmd_name} already registered')

        self._commands[cmd_name] = cmd
        func = functools.partial(self.run, cmd_name)
        setattr(self, cmd_name, func)

    def run(self, command_name, **kwargs):
        # use registered command
        if command_name in self._commands:
            function = self._commands[command_name]
            function_signature = inspect.signature(function)

            # user specified an object name
            object_handle = None
            if 'name' in kwargs:
                object_handle = kwargs.pop('name')

            # error if an argument is not matching the function signature
            for args in kwargs:
                if not args in function_signature.parameters:
                    raise ValueError("The argument '" + args +
                                     "' doesn't match the function signature from '"  + command_name + "'")

            # prepare function arguments
            function_args = {}
            for param_name in function_signature.parameters:
                #param_obj = function_signature.parameters[param_name]
                param_value = self._convert_parameter(param_name, kwargs)
                if param_value is not None:
                    function_args[param_name] = param_value

            # call function
            function_result = function(**function_args)
            return self._process_result(function_result, object_handle)

        raise ValueError("The command  " + command_name + " is not recognized.'")

    def _convert_parameter(self, parameter_name, kwargs):
        # parameter provided by user
        if parameter_name in kwargs:
            return kwargs[parameter_name]
        return None

    def _process_result(self, result, object_handle=None):
        return result



