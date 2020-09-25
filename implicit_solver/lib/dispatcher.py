"""
@author: Vincent Bonnet
@description : command dispatcher for solver
"""

# import for CommandSolverDispatcher
import uuid

from core import Details
import lib.system as system
import lib.system.time_integrators as integrator
from lib.objects import Dynamic, Kinematic, Condition, Force
from lib.objects.jit.data import Node, Spring, AnchorSpring, Bending, Area
from lib.objects.jit.data import Point, Edge, Triangle
import lib.objects.commands as cmd
import core

class CommandSolverDispatcher(core.CommandDispatcher):
    '''
    Dispatch commands to manage objects (animators, conditions, dynamics, kinematics, forces)
    '''
    def __init__(self):
        core.CommandDispatcher.__init__(self)
        # data
        self._scene = None
        self._details = None
        self._reset()
        self._solver = system.Solver(integrator.BackwardEulerIntegrator())
        #self._solver = system.Solver(integrator.SymplecticEulerIntegrator())
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
        self.register_cmd(cmd.initialize)
        self.register_cmd(cmd.add_dynamic)
        self.register_cmd(cmd.add_kinematic)
        self.register_cmd(cmd.solve_to_next_frame)
        self.register_cmd(cmd.get_nodes_from_dynamic)
        self.register_cmd(cmd.get_shape_from_kinematic)
        self.register_cmd(cmd.get_normals_from_kinematic)
        self.register_cmd(cmd.get_segments_from_constraint)
        self.register_cmd(cmd.set_render_prefs)
        self.register_cmd(cmd.add_gravity)
        self.register_cmd(cmd.add_edge_constraint)
        self.register_cmd(cmd.add_wire_bending_constraint)
        self.register_cmd(cmd.add_face_constraint)
        self.register_cmd(cmd.add_kinematic_attachment)
        self.register_cmd(cmd.add_kinematic_collision)
        self.register_cmd(cmd.add_dynamic_attachment)
        self.register_cmd(cmd.get_sparse_matrix_as_dense)

    def _add_object(self, obj, object_handle=None):
        if object_handle in self._object_dict:
            assert False, f'_add_object(...) {object_handle} already exists'

        if not object_handle:
            object_handle = str(uuid.uuid4())

        if isinstance(obj, (Dynamic, Kinematic, Condition, Force)):
            self._object_dict[object_handle] = obj
        else:
            assert False, '_add_object(...) only supports Dynamic, Kinematic, Condition and Force'

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
        if isinstance(result, (Dynamic, Kinematic, Condition, Force)):
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
        system_types = [Node, Area, Bending, Spring, AnchorSpring]
        system_types += [Point, Edge, Triangle]
        group_types = {'dynamics' : [Node],
                       'constraints' : [Area, Bending, Spring, AnchorSpring],
                       'geometries': [Point, Edge, Triangle],
                       'bundle': system_types}
        self._details = Details(system_types, group_types)
