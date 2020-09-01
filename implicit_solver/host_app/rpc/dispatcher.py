"""
@author: Vincent Bonnet
@description : Run command and bundle scene/solver/context together
"""

# import for CommandDispatcher
import functools
import inspect

# import for CommandSolverDispatcher
import uuid
import lib.system as system
import lib.system.time_integrators as integrator
import lib.objects as lib_objects
import logic

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


class CommandSolverDispatcher(CommandDispatcher):
    '''
    Dispatch commands to manage objects (animators, conditions, dynamics, kinematics, forces)
    '''
    def __init__(self):
        CommandDispatcher.__init__(self)
        # data
        self._scene = system.Scene()
        self._solver = system.Solver(integrator.BackwardEulerIntegrator())
        #self._solver = system.Solver(integrator.SymplecticEulerIntegrator())
        self._details = system.SolverDetails()
        self._context = system.SolverContext()
        # map hash_value with objects (dynamic, kinematic, condition, force)
        self._object_dict = {}

        # register
        self.register_cmd(self._set_context, 'set_context')
        self.register_cmd(self._get_context, 'get_context')
        self.register_cmd(self._get_dynamics, 'get_dynamics')
        self.register_cmd(self._get_conditions, 'get_conditions')
        self.register_cmd(self._get_kinematics, 'get_kinematics')
        self.register_cmd(self._get_metadata, 'get_metadata')
        self.register_cmd(self._get_commands, 'get_commands')
        self.register_cmd(self._reset, 'reset')
        self.register_cmd(logic.initialize)
        self.register_cmd(logic.add_dynamic)
        self.register_cmd(logic.add_kinematic)
        self.register_cmd(logic.solve_to_next_frame)
        self.register_cmd(logic.get_nodes_from_dynamic)
        self.register_cmd(logic.get_shape_from_kinematic)
        self.register_cmd(logic.get_normals_from_kinematic)
        self.register_cmd(logic.get_segments_from_constraint)
        self.register_cmd(logic.set_render_prefs)
        self.register_cmd(logic.add_gravity)
        self.register_cmd(logic.add_edge_constraint)
        self.register_cmd(logic.add_wire_bending_constraint)
        self.register_cmd(logic.add_face_constraint)
        self.register_cmd(logic.add_kinematic_attachment)
        self.register_cmd(logic.add_kinematic_collision)
        self.register_cmd(logic.add_dynamic_attachment)
        self.register_cmd(logic.get_sparse_matrix_as_dense)

    def _add_object(self, obj, object_handle=None):
        if object_handle in self._object_dict:
            assert False, f'_add_object(...) {object_handle} already exists'

        if not object_handle:
            object_handle = str(uuid.uuid4())

        if isinstance(obj, lib_objects.Dynamic):
            self._object_dict[object_handle] = obj
        elif isinstance(obj, lib_objects.Kinematic):
            self._object_dict[object_handle] = obj
        elif isinstance(obj, lib_objects.Condition):
            self._object_dict[object_handle] = obj
        elif isinstance(obj, lib_objects.Force):
            self._object_dict[object_handle] = obj
        else:
            assert False, '_add_object(...) only supports lib.Dynamic, lib.Kinematic, lib.Condition and lib.Force'

        return object_handle

    def _convert_parameter(self, parameter_name, kwargs):
        # parameter provided by the dispatcher
        if parameter_name == 'scene':
            return self._scene
        elif parameter_name == 'solver':
            return self._solver
        elif parameter_name == 'context':
            return self._context
        elif parameter_name == 'details':
            return self._details

        # parameter provided by user
        if parameter_name in kwargs:
            arg_object = kwargs[parameter_name]
            reserved_attrs = ['dynamic','kinematic','condition','obj']
            is_reserved_attr = False
            for reserved_attr in reserved_attrs:
                if not parameter_name.startswith(reserved_attr):
                    continue
                is_reserved_attr = True
                break

            if is_reserved_attr:
                if arg_object not in self._object_dict:
                    assert False, f'in _convert_parameter(...) {arg_object} doesnt exist'
                return self._object_dict[arg_object]

            return kwargs[parameter_name]

        return None

    def _process_result(self, result, object_handle=None):
        # convert the result object
        if isinstance(result, (lib_objects.Dynamic,
                               lib_objects.Kinematic,
                               lib_objects.Condition,
                               lib_objects.Force)):
            # the object is already stored
            for k, v in self._object_dict.items():
                if v == result:
                    return k

            # add the new object
            return self._add_object(result, object_handle)

        if isinstance(result, (tuple, list)):
            # shallow copy to not override the original list
            result = result.copy()
            for index in range(len(result)):
                result[index] = self._process_result(result[index])

        return result

    def _set_context(self, time : float, frame_dt : float, num_substep : int, num_frames : int):
        self._context = system.SolverContext(time, frame_dt, num_substep, num_frames)

    def _get_context(self):
        return self._context

    def _get_dynamics(self):
        return self._scene.dynamics

    def _get_conditions(self):
        return self._scene.conditions

    def _get_kinematics(self):
        return self._scene.kinematics

    def _get_metadata(self, obj):
        if obj:
            return obj.metadata()
        return None

    def _get_commands(self):
        return list(self._commands.keys())

    def _reset(self):
        self._scene = system.Scene()
        self._details = system.SolverDetails()

