"""
@author: Vincent Bonnet
@description : Run command and bundle scene/solver/context together
"""

# TODO - future work
# dispatcher should not be aware of the solver libraries (lib.*)
import lib.system as system
import lib.objects as lib_objects
import uuid
import inspect

class CommandDispatcher:
    '''
    Dispatch commands to manage objects (animators, conditions, dynamics, kinematics, forces)
    '''
    def __init__(self):
        # data
        self._scene = system.Scene()
        self._solver = system.Solver(system.ImplicitSolver())
        self._context = system.Context()
        # map hash_value with object
        self._object_dict = {}
        # list of registered commands
        self._commands = {}

    def is_defined(self):
        if self._scene and self._solver and self._context:
            return True
        return False

    def __add_object(self, obj):
        unique_id = uuid.uuid4()
        self._object_dict[unique_id] = obj
        return unique_id

    def register_cmd(self, cmd):
        self._commands[cmd.__name__] = cmd

    def __convert_parameter(self, parameter_name, kwargs):
        # parameter provided by user
        if parameter_name in kwargs:
            arg_object = kwargs[parameter_name]
            if isinstance(arg_object, uuid.UUID):
                return self._object_dict[arg_object]

            return kwargs[parameter_name]

        # parameter provided by the dispatcher
        if parameter_name == 'scene':
            return self._scene
        elif parameter_name == 'solver':
            return self._solver
        elif parameter_name == 'context':
            return self._context

        return None

    def __process_result(self, result):
        # convert the result object
        if isinstance(result, (lib_objects.Force,
                               lib_objects.Dynamic,
                               lib_objects.Kinematic,
                               lib_objects.Condition)):
            return self.__add_object(result)

        return result

    def run(self, command_name, **kwargs):
        '''
        Execute functions from system.commands
        '''
        # use registered command
        if command_name in self._commands:
            function = self._commands[command_name]
            function_signature = inspect.signature(function)
            function_args = {}
            # TODO - error if any kwargs not matching function_signature.parameters
            for param_name in function_signature.parameters:
                #param_obj = function_signature.parameters[param_name]
                param_value = self.__convert_parameter(param_name, kwargs)
                if param_value is not None:
                    function_args[param_name] = param_value

            function_result = function(**function_args)
            return self.__process_result(function_result)

        # TODO : hardcoded - to be removed
        result = None
        if (command_name == 'set_context'):
            self._context = kwargs['context']
            result = True
        elif (command_name == 'get_context'):
            result = self._context
        elif (command_name == 'get_scene'):
            result = self._scene
        elif (command_name == 'get_dynamic_handles'):
            result = []
            for obj in self._scene.dynamics:
                for k, v in self._object_dict.items():
                    if obj == v:
                        result.append(k)
        elif (command_name == 'reset_scene'):
            self._scene = system.Scene()
        else:
            raise ValueError("The command  " + command_name + " is not recognized.'")

        return result
