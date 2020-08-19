"""
@author: Vincent Bonnet
@description : Subclasses of the Condition class
"""

import numba
import numpy as np

from lib.objects import Condition
from lib.objects.jit.data import Node, AnchorSpring, Spring, Area, Bending
import lib.objects.jit.algorithms as algo
import lib.common.jit.block_utils as block_utils
import lib.common.code_gen as generate

def initialize_condition_from_aos(condition, array_of_struct, details):
    data = details.block_from_datatype(condition.typename)

    # disable previous allocated blocks
    num_constraints = len(array_of_struct)
    condition.total_constraints = num_constraints

    data.set_active(False, condition.block_handles)
    condition.block_handles = block_utils.empty_block_handles()

    # early exit if there is no constraints
    if (num_constraints == 0):
        return False

    # allocate
    block_handles = data.append_empty(num_constraints, reuse_inactive_block=True)
    condition.block_handles = block_handles

    # copy to datablock
    num_elements = len(array_of_struct)
    for field_name, value in array_of_struct[0].__dict__.items():
        # create contiguous array
        data_type = None
        data_shape = None
        if np.isscalar(value):
            # (num_elements, ) guarantees to be array and not single value
            data_type = type(value)
            data_shape = (num_elements,)
        else:
            data_type = value.dtype.type
            list_shape = list(value.shape)
            list_shape.insert(0, num_elements)
            data_shape = tuple(list_shape)

        new_array = np.zeros(shape=data_shape, dtype=data_type)

        # set contiguous array
        for element_id, element in enumerate(array_of_struct):
            new_array[element_id] = getattr(element, field_name)

        # set datbablock
        data.copyto(field_name, new_array, condition.block_handles)

    # compute constraint rest
    condition.compute_rest(details.bundle)

    return True


@generate.vectorize(njit=True)
def appendKinematicCollision(node : Node, points, edges, triangles, edge_handles, triangle_handles, is_inside_func, closest_param_func):
    result = algo.simplex_lib.IsInsideResult()
    result.isInside = False
    is_inside_func(triangles, points, node.x, result, triangle_handles)
    if (result.isInside):
        closest_param = algo.simplex_lib.ClosestResult()
        closest_param_func(edges, points, node.x, closest_param, edge_handles)

        if np.dot(closest_param.normal, node.v) < 0.0:
            pass
            # TODO - before adding contact, remove those outside the stuff !
            # TODO - add contacts - details.spring
            #print('ADD contacts')
            #block_id = 0
            #data = spring_data[block_id]
            #data0 = np.empty_like(data)


class KinematicCollisionCondition(Condition):
    '''
    Creates collision constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping):
        Condition.__init__(self, stiffness, damping, AnchorSpring)
        # data
        self.dynamic_handles = dynamic.block_handles
        self.triangle_handles = kinematic.triangle_handles
        self.edge_handles = kinematic.edge_handles
        # functions
        self.func.pre_compute = algo.anchor_spring_lib.pre_compute
        self.func.compute_rest = algo.anchor_spring_lib.compute_rest
        self.func.compute_function = None
        self.func.compute_gradients = None
        self.func.compute_hessians = None
        self.func.compute_forces = algo.anchor_spring_lib.compute_forces
        self.func.compute_force_jacobians = algo.anchor_spring_lib.compute_force_jacobians

    def is_static(self):
        '''
        Returns False because collision constraints are dynamics
        '''
        return False

    def init_constraints(self, details):
        '''
        Add zero-length springs into anchor spring details
        '''
        '''
        appendKinematicCollision(details.node,
                                 details.point,
                                 details.edge,
                                 details.triangle,
                                 self.edge_handles,
                                 self.triangle_handles,
                                 simplex_lib.is_inside.function,
                                 simplex_lib.get_closest_param.function,
                                 self.dynamic_handles)
        '''

        springs = []

        data_x = details.node.flatten('x', self.dynamic_handles)
        data_v = details.node.flatten('v', self.dynamic_handles)
        data_node_id = details.node.flatten('ID', self.dynamic_handles)

        result = algo.simplex_lib.IsInsideResult()
        for i in range(len(data_x)):
            node_pos = data_x[i]
            node_vel = data_v[i]
            node_ids = [data_node_id[i]]

            result.isInside = False
            algo.simplex_lib.is_inside(details.triangle,
                                       details.point,
                                       node_pos,
                                       result,
                                       self.triangle_handles)

            if (result.isInside):
                closest_param = algo.simplex_lib.ClosestResult()
                algo.simplex_lib.get_closest_param(details.edge,
                                                   details.point, node_pos,
                                                   closest_param,
                                                   self.edge_handles)

                if (np.dot(closest_param.normal, node_vel) < 0.0):
                    # add spring
                    spring = AnchorSpring()
                    spring.kinematic_component_IDs =  closest_param.points
                    spring.kinematic_component_param = np.float64(closest_param.t)
                    spring.kinematic_component_pos = closest_param.position
                    spring.node_IDs = np.copy(node_ids)
                    spring.stiffness = self.stiffness
                    spring.damping = self.damping
                    springs.append(spring)

        initialize_condition_from_aos(self, springs, details)



    def update_constraints(self, details):
        self.init_constraints(details)

class KinematicAttachmentCondition(Condition):
    '''
    Creates attachment constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping, distance):
        Condition.__init__(self, stiffness, damping, AnchorSpring)
        # data
        self.distance = distance
        self.dynamic_handles = dynamic.block_handles
        self.edge_handles = kinematic.edge_handles
        # functions
        self.func.pre_compute = algo.anchor_spring_lib.pre_compute
        self.func.compute_rest = algo.anchor_spring_lib.compute_rest
        self.func.compute_function = None
        self.func.compute_gradients = None
        self.func.compute_hessians = None
        self.func.compute_forces = algo.anchor_spring_lib.compute_forces
        self.func.compute_force_jacobians = algo.anchor_spring_lib.compute_force_jacobians

    def init_constraints(self, details):
        '''
        Add springs into the anchor spring details
        '''
        springs = []

        data_x = details.node.flatten('x', self.dynamic_handles)
        data_node_id = details.node.flatten('ID', self.dynamic_handles)

        # Linear search => it will be inefficient for dynamic objects with many nodes
        distance2 = self.distance * self.distance
        for i in range(len(data_x)):
            node_pos = data_x[i]
            node_ids = [data_node_id[i]]

            closest_param = algo.simplex_lib.ClosestResult()
            algo.simplex_lib.get_closest_param(details.edge,
                                               details.point, node_pos,
                                               closest_param,
                                               self.edge_handles)

            if closest_param.squared_distance < distance2:
                # add spring
                spring = AnchorSpring()
                spring.kinematic_component_IDs = closest_param.points
                spring.kinematic_component_param = np.float64(closest_param.t)
                spring.kinematic_component_pos = closest_param.position
                spring.node_IDs = np.copy(node_ids)
                spring.stiffness = self.stiffness
                spring.damping = self.damping
                springs.append(spring)

        initialize_condition_from_aos(self, springs, details)


class DynamicAttachmentCondition(Condition):
    '''
    Creates attachment constraint between two dynamic objects
    '''
    def __init__(self, dynamic0, dynamic1, stiffness, damping, distance):
       Condition.__init__(self, stiffness, damping, Spring)
       self.distance = distance
       # data
       self.dynamic0_handles = dynamic0.block_handles
       self.dynamic1_handles = dynamic1.block_handles
       # functions
       self.func.pre_compute = None
       self.func.compute_rest = algo.spring_lib.compute_rest
       self.func.compute_function = None
       self.func.compute_gradients = None
       self.func.compute_hessians = None
       self.func.compute_forces = algo.spring_lib.compute_forces
       self.func.compute_force_jacobians = algo.spring_lib.compute_force_jacobians

    def init_constraints(self, details):
        '''
        Add springs into the spring details
        '''
        springs = []
        distance2 = self.distance * self.distance

        data_x0 = details.node.flatten('x', self.dynamic0_handles)
        data_node_id0 = details.node.flatten('ID', self.dynamic0_handles)
        data_x1 = details.node.flatten('x', self.dynamic1_handles)
        data_node_id1 = details.node.flatten('ID', self.dynamic1_handles)

        for i in range(len(data_x0)):
            for j in range(len(data_x1)):
                x0 = data_x0[i]
                x1 = data_x1[j]
                direction = (x0 - x1)
                dist2 = np.inner(direction, direction)
                if dist2 < distance2:
                    node_id0 = data_node_id0[i]
                    node_id1 = data_node_id1[j]
                    node_ids = [node_id0, node_id1]

                    # add spring
                    spring = Spring()
                    spring.node_IDs = np.copy(node_ids)
                    spring.stiffness = self.stiffness
                    spring.damping = self.damping
                    springs.append(spring)

        initialize_condition_from_aos(self, springs, details)

class EdgeCondition(Condition):
    '''
    Creates Spring constraints
    Replaces edges with Spring constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, stiffness, damping, Spring)
       self.dynamics = dynamics.copy()
       # functions
       self.func.pre_compute = None
       self.func.compute_rest = algo.spring_lib.compute_rest
       self.func.compute_function = None
       self.func.compute_gradients = None
       self.func.compute_hessians = None
       self.func.compute_forces = algo.spring_lib.compute_forces
       self.func.compute_force_jacobians = algo.spring_lib.compute_force_jacobians

    def init_constraints(self, details):
        springs = []
        for dynamic in self.dynamics:
            for vertex_index in dynamic.edge_ids:
                node_ids = [None, None]
                node_ids[0] = dynamic.get_node_id(vertex_index[0])
                node_ids[1] = dynamic.get_node_id(vertex_index[1])

                # add spring
                spring = Spring()
                spring.node_IDs = np.copy(node_ids)
                spring.stiffness = self.stiffness
                spring.damping = self.damping
                springs.append(spring)

        initialize_condition_from_aos(self, springs, details)

class AreaCondition(Condition):
    '''
    Creates Area constraints
    Replaces triangle with Area constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, stiffness, damping, Area)
       self.dynamics = dynamics.copy()
       # functions
       self.func.pre_compute = None
       self.func.compute_rest = algo.area_lib.compute_rest
       self.func.compute_function = None
       self.func.compute_gradients = None
       self.func.compute_hessians = None
       self.func.compute_forces = algo.area_lib.compute_forces
       self.func.compute_force_jacobians = algo.area_lib.compute_force_jacobians

    def init_constraints(self, details):
        constraints = []

        for dynamic in self.dynamics:
            for vertex_index in dynamic.face_ids:
                node_ids = [None, None, None]
                node_ids[0] = dynamic.get_node_id(vertex_index[0])
                node_ids[1] = dynamic.get_node_id(vertex_index[1])
                node_ids[2] = dynamic.get_node_id(vertex_index[2])

                # add area constraint
                constraint = Area()
                constraint.node_IDs = np.copy(node_ids)
                constraint.stiffness = self.stiffness
                constraint.damping = self.damping
                constraints.append(constraint)

        initialize_condition_from_aos(self, constraints, details)

class WireBendingCondition(Condition):
    '''
    Creates Wire Bending constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, stiffness, damping, Bending)
       self.dynamics = dynamics.copy()
       self.node_ids = []
       for dynamic in self.dynamics:
           # create a dictionnary of neighbour
           vtx_neighbours = {}
           for vtx_ids in dynamic.edge_ids:
               vtx_neighbours.setdefault(vtx_ids[0], []).append(vtx_ids[1])
               vtx_neighbours.setdefault(vtx_ids[1], []).append(vtx_ids[0])
           # create the node_ids
           for vtx_index, vtx_neighbour_index in vtx_neighbours.items():
               if (len(vtx_neighbour_index) == 2):
                        node_ids = [dynamic.get_node_id(vtx_neighbour_index[0]),
                                    dynamic.get_node_id(vtx_index),
                                    dynamic.get_node_id(vtx_neighbour_index[1])]
                        self.node_ids.append(node_ids)

       self.node_ids = np.asarray(self.node_ids)
       # functions
       self.func.pre_compute = None
       self.func.compute_rest = algo.bending_lib.compute_rest
       self.func.compute_function = None
       self.func.compute_gradients = None
       self.func.compute_hessians = None
       self.func.compute_forces = algo.bending_lib.compute_forces
       self.func.compute_force_jacobians = algo.bending_lib.compute_force_jacobians

    def init_constraints(self, details):
        constraints = []

        for node_ids in self.node_ids:
            # add bending constraint
            constraint = Bending()
            constraint.node_IDs = np.copy(node_ids)
            constraint.stiffness = self.stiffness
            constraint.damping = self.damping
            constraints.append(constraint)

        initialize_condition_from_aos(self, constraints, details)
