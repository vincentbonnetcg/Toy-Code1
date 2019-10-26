"""
@author: Vincent Bonnet
@description : Run command and bundle scene/solver/context together
"""

# import for CommandDispatcher
import inspect

# import for CommandSolverDispatcher
import uuid
import lib.system as system
import lib.objects as lib_objects
import logic.commands_lib as sim_cmds
import logic.commands_subclass as subclass_cmds

class CommandDispatcher:
    '''
    Base class to developer command dispatcher
    '''
    def __init__(self):
        # list of registered commands
        self._commands = {}

    def register_cmd(self, cmd):
        self._commands[cmd.__name__] = cmd

    def run(self, command_name, **kwargs):
        # use registered command
        if command_name in self._commands:
            function = self._commands[command_name]
            function_signature = inspect.signature(function)
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
            return self._process_result(function_result)

        raise ValueError("The command  " + command_name + " is not recognized.'")

    def _convert_parameter(self, parameter_name, kwargs):
        # parameter provided by user
        if parameter_name in kwargs:
            return kwargs[parameter_name]
        return None

    def _process_result(self, result):
        return result


class CommandSolverDispatcher(CommandDispatcher):
    '''
    Dispatch commands to manage objects (animators, conditions, dynamics, kinematics, forces)
    '''
    def __init__(self):
        CommandDispatcher.__init__(self)
        # data
        self._scene = system.Scene()
        self._solver = system.Solver(system.ImplicitSolver())
        self._context = system.Context()
        # map hash_value with objects (dynamic, kinematic, condition, force)
        self._object_dict = {}
        self._dynamic_handles = {}
        self._kinematic_handles = {}
        self._conditions_handles = {}
        self._force_handles = {}

        # register
        self.register_cmd(self.set_context)
        self.register_cmd(self.get_context)
        self.register_cmd(self.get_scene)
        self.register_cmd(self.get_dynamic_handles)
        self.register_cmd(self.reset_scene)
        self.register_cmd(sim_cmds.initialize)
        self.register_cmd(sim_cmds.add_dynamic)
        self.register_cmd(sim_cmds.add_kinematic)
        self.register_cmd(sim_cmds.solve_to_next_frame)
        self.register_cmd(sim_cmds.set_render_prefs)
        self.register_cmd(subclass_cmds.add_gravity)
        self.register_cmd(subclass_cmds.add_edge_constraint)
        self.register_cmd(subclass_cmds.add_wire_bending_constraint)
        self.register_cmd(subclass_cmds.add_face_constraint)
        self.register_cmd(subclass_cmds.add_kinematic_attachment)
        self.register_cmd(subclass_cmds.add_kinematic_collision)
        self.register_cmd(subclass_cmds.add_dynamic_attachment)

    def _add_object(self, obj):
        unique_id = uuid.uuid4()

        if isinstance(obj, lib_objects.Dynamic):
            self._dynamic_handles[unique_id] = obj
            self._object_dict[unique_id] = obj
        elif isinstance(obj, lib_objects.Kinematic):
            self._kinematic_handles[unique_id] = obj
            self._object_dict[unique_id] = obj
        elif isinstance(obj, lib_objects.Condition):
            self._conditions_handles[unique_id] = obj
            self._object_dict[unique_id] = obj
        elif isinstance(obj, lib_objects.Force):
            self._force_handles[unique_id] = obj
            self._object_dict[unique_id] = obj
        else:
            assert False, '_add_object(..) only supports lib.Dynamic, lib.Kinematic, lib.Condition and lib.Force'

        return unique_id

    def _convert_parameter(self, parameter_name, kwargs):
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

    def _process_result(self, result):
        # convert the result object
        if isinstance(result, (lib_objects.Dynamic,
                               lib_objects.Kinematic,
                               lib_objects.Condition,
                               lib_objects.Force)):
            return self._add_object(result)

        return result

    def set_context(self, context):
        self._context = context

    def get_context(self):
        return self._context

    def get_scene(self):
        return self._scene

    def get_dynamic_handles(self):
        handles = []
        for handle in self._dynamic_handles:
            handles.append(handle)
        return handles

    def reset_scene(self):
        self._scene = system.Scene()
