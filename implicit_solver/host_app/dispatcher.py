"""
@author: Vincent Bonnet
@description : Run command and bundle scene/solver/context together
"""

import system
import system.commands as sim_cmds
import system.setup.commands as setup_cmds
import uuid

class CommandDispatcher:
    '''
    Dispatch commands to manage objects (animators, conditions, dynamics, kinematics, forces)
    '''
    def __init__(self, context = None):
        self._scene = system.Scene()
        self._solver = system.ImplicitSolver()
        self._context = context
        self._object_dict = {} # map hash_value with object

    def is_defined(self):
        if self._scene and self._solver and self._context:
            return True
        return False

    def __add_object(self, obj):
        unique_id = uuid.uuid4()
        self._object_dict[unique_id] = obj
        return unique_id

    def run(self, command_name, *args):
        '''
        Execute functions from system.setup.commands and system.commands
        '''
        result = None

        dispatch = {'initialize' : sim_cmds.initialize,
                    'solve_to_next_frame' : sim_cmds.solve_to_next_frame,
                    'add_dynamic' : setup_cmds.add_dynamic,
                    'add_kinematic' : setup_cmds.add_kinematic,
                    'add_edge_constraint' : setup_cmds.add_edge_constraint,
                    'add_wire_bending_constraint' : setup_cmds.add_wire_bending_constraint,
                    'add_kinematic_attachment' : setup_cmds.add_kinematic_attachment,
                    'add_kinematic_collision' : setup_cmds.add_kinematic_collision,
                    'add_gravity_acceleration' : setup_cmds.add_gravity_acceleration,
                    'add_render_prefs' : setup_cmds.add_render_prefs}

        if (command_name == 'initialize' or
            command_name == 'solve_to_next_frame'):
            dispatch[command_name](self._scene, self._solver, self._context)
        elif (command_name == 'add_dynamic' or
              command_name == 'add_kinematic'):
            new_obj = dispatch[command_name](self._scene, *args)
            result = uuid.uuid4() # object hash
            self._object_dict[result] = new_obj
        elif (command_name == 'add_edge_constraint' or
              command_name == 'add_wire_bending_constraint'):
            obj = self._object_dict[args[0]]
            new_obj = dispatch[command_name](self._scene, obj, args[1], args[2])
            result = self.__add_object(new_obj)
        elif (command_name == 'add_kinematic_attachment'):
            dyn_obj = self._object_dict[args[0]]
            kin_obj = self._object_dict[args[1]]
            new_obj = dispatch[command_name](self._scene, dyn_obj, kin_obj, args[2], args[3], args[4])
            result = self.__add_object(new_obj)
        elif (command_name == 'add_kinematic_collision'):
            dyn_obj = self._object_dict[args[0]]
            kin_obj = self._object_dict[args[1]]
            new_obj = dispatch[command_name](self._scene, dyn_obj, kin_obj, args[2], args[3])
            result = self.__add_object(new_obj)
        elif (command_name == 'add_gravity_acceleration'):
            new_obj = dispatch[command_name](self._scene, args[0])
            result = self.__add_object(new_obj)
        elif (command_name == 'add_render_prefs'):
            obj = self._object_dict[args[0]]
            dispatch[command_name](obj, args[1])
        else:
            print("The command  " + command_name + " is not recognized !")

        return result

    def scene(self):
        return self._scene

    def solver(self):
        return self._solver

    def context(self):
        return self._context

