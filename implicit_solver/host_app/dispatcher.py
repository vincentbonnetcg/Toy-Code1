"""
@author: Vincent Bonnet
@description : Run command and bundle scene/solver/context together
"""

import system
import system.commands as sim_cmds
import tests.commands as subclass_cmds
import uuid

class CommandDispatcher:
    '''
    Dispatch commands to manage objects (animators, conditions, dynamics, kinematics, forces)
    '''
    def __init__(self):
        self._scene = system.Scene()
        self._solver = system.ImplicitSolver()
        self._context = system.Context()
        self._object_dict = {} # map hash_value with object

    def is_defined(self):
        if self._scene and self._solver and self._context:
            return True
        return False

    def __add_object(self, obj):
        unique_id = uuid.uuid4()
        self._object_dict[unique_id] = obj
        return unique_id

    def run(self, command_name, **kwargs):
        '''
        Execute functions from system.commands
        '''
        result = None

        dispatch = {'set_context' : None,
                    'get_context' : None,
                    'get_scene' : None,
                    'initialize' : sim_cmds.initialize,
                    'reset_scene' : None,
                    'solve_to_next_frame' : sim_cmds.solve_to_next_frame,
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
        if (command_name == 'set_context'):
            self._context = kwargs['context']
            result = True
        if (command_name == 'get_context'):
            result = self._context
        elif (command_name == 'get_scene'):
            result = self._scene
        elif (command_name == 'reset_scene'):
            self._scene = system.Scene()
        elif (command_name == 'initialize' or
              command_name == 'solve_to_next_frame'):
            dispatch[command_name](self._scene, self._solver, self._context)
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
            assert("The command  " + command_name + " is not recognized !")

        return result
