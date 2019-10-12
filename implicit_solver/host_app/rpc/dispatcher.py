"""
@author: Vincent Bonnet
@description : Run command and bundle scene/solver/context together
"""

import lib.system as system
import logic.commands_lib as sim_cmds
import logic.commands_subclass as subclass_cmds
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

    def __convert_parameter(self, parameter_name):
        if parameter_name == 'scene':
            return self._scene
        elif parameter_name == 'solver':
            return self._solver
        elif parameter_name == 'context':
            return self._context

        raise ValueError("The parameter  " + parameter_name + " is not recognized.'")

        return None


    def run(self, command_name, **kwargs):
        '''
        Execute functions from system.commands
        '''
        result = None

        dispatch = {'set_context' : None,
                    'get_context' : None,
                    'get_scene' : None,
                    'get_dynamic_handles' : None,
                    'reset_scene' : None,
                    'add_dynamic' : sim_cmds.add_dynamic,
                    'add_kinematic' : sim_cmds.add_kinematic,
                    'add_edge_constraint' : subclass_cmds.add_edge_constraint,
                    'add_wire_bending_constraint' : subclass_cmds.add_wire_bending_constraint,
                    'add_face_constraint': subclass_cmds.add_face_constraint,
                    'add_kinematic_attachment' : subclass_cmds.add_kinematic_attachment,
                    'add_kinematic_collision' : subclass_cmds.add_kinematic_collision,
                    'add_dynamic_attachment' : subclass_cmds.add_dynamic_attachment,
                    'add_gravity' : subclass_cmds.add_gravity,
                    'set_render_prefs' : sim_cmds.set_render_prefs}
        # TODO : hardcoded - to be removed
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
        elif (command_name == 'add_dynamic' or
              command_name == 'add_kinematic'):
            new_obj = dispatch[command_name](self._scene, **kwargs)
            result = self.__add_object(new_obj)
        elif (command_name == 'add_edge_constraint' or
              command_name == 'add_wire_bending_constraint' or
              command_name == 'add_face_constraint'):
            obj = self._object_dict[kwargs['dynamic']]
            new_obj = dispatch[command_name](self._scene, obj, kwargs['stiffness'], kwargs['damping'])
            result = self.__add_object(new_obj)
        elif (command_name == 'add_kinematic_attachment'):
            dyn_obj = self._object_dict[kwargs['dynamic']]
            kin_obj = self._object_dict[kwargs['kinematic']]
            new_obj = dispatch[command_name](self._scene, dyn_obj, kin_obj,
                                                          kwargs['stiffness'],
                                                          kwargs['damping'],
                                                          kwargs['distance'])
            result = self.__add_object(new_obj)
        elif (command_name == 'add_dynamic_attachment'):
            dyn0_obj = self._object_dict[kwargs['dynamic0']]
            dyn1_obj = self._object_dict[kwargs['dynamic1']]
            new_obj = dispatch[command_name](self._scene, dyn0_obj, dyn1_obj,
                                                          kwargs['stiffness'],
                                                          kwargs['damping'],
                                                          kwargs['distance'])
        elif (command_name == 'add_kinematic_collision'):
            dyn_obj = self._object_dict[kwargs['dynamic']]
            kin_obj = self._object_dict[kwargs['kinematic']]
            new_obj = dispatch[command_name](self._scene, dyn_obj, kin_obj,
                                                          kwargs['stiffness'],
                                                          kwargs['damping'])
            result = self.__add_object(new_obj)
        elif (command_name == 'add_gravity'):
            new_obj = dispatch[command_name](self._scene, kwargs['gravity'])
            result = self.__add_object(new_obj)
        elif (command_name == 'set_render_prefs'):
            obj = self._object_dict[kwargs['obj']]
            dispatch[command_name](obj, kwargs['prefs'])
        else:
            # use registered command
            # TODO : for now only support (scene, solver, context)
            if command_name in self._commands:
                function = self._commands[command_name]
                function_signature = inspect.signature(function)
                function_args = {}
                for param_name in function_signature.parameters:
                    #param_obj = function_signature.parameters[param_name]
                    param_value = self.__convert_parameter(param_name)
                    function_args[param_name] = param_value

                function(**function_args)
            else:
                raise ValueError("The command  " + command_name + " is not recognized.'")

        return result
