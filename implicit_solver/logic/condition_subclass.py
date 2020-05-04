"""
@author: Vincent Bonnet
@description : Subclasses of the Condition class
"""

import numpy as np

import lib.objects.jit as cpn
from lib.objects import Condition
import lib.common as common
import lib.common.jit.block_utils as block_utils
from lib.system import Scene

def initialize_condition_from_aos(condition, array_of_struct, details):
    data = details.block_from_datatype(condition.constraint_type)

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
    condition.compute_rest(details)

    return True


class KinematicCollisionCondition(Condition):
    '''
    Creates collision constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping):
        Condition.__init__(self, stiffness, damping, cpn.AnchorSpring)
        self.dynamic_indices = [dynamic.index]
        self.kinematic_indices = [kinematic.index]

    def is_static(self):
        '''
        Returns False because collision constraints are dynamics
        '''
        return False

    def init_constraints(self, scene : Scene, details):
        '''
        Add zero-length springs into the dynamic constraints of the scene
        '''
        dynamic = scene.dynamics[self.dynamic_indices[0]]
        kinematic = scene.kinematics[self.kinematic_indices[0]]
        springs = []

        data_x = details.node.flatten('x', dynamic.block_handles)
        data_v = details.node.flatten('v', dynamic.block_handles)
        data_node_id = details.node.flatten('ID', dynamic.block_handles)

        for i in range(dynamic.num_nodes()):
            node_pos = data_x[i]
            node_vel = data_v[i]
            node_ids = [data_node_id[i]]

            if (kinematic.is_inside(node_pos)):
                closest_pos = kinematic.get_closest_param(details, node_pos)
                if (np.dot(closest_pos.normal, node_vel) < 0.0):
                    # add spring
                    spring = cpn.AnchorSpring()
                    spring.kinematic_index = np.uint32(kinematic.index)
                    spring.kinematic_component_index =  np.uint32(closest_pos.index)
                    spring.kinematic_component_param = np.float64(closest_pos.t)
                    spring.kinematic_component_pos = closest_pos.position
                    spring.node_IDs = np.copy(node_ids)
                    spring.stiffness = self.stiffness
                    spring.damping = self.damping
                    springs.append(spring)

        initialize_condition_from_aos(self, springs, details)

    def update_constraints(self, scene : Scene, details):
        self.init_constraints(scene, details)

class KinematicAttachmentCondition(Condition):
    '''
    Creates attachment constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping, distance):
       Condition.__init__(self, stiffness, damping, cpn.AnchorSpring)
       self.distance = distance
       self.dynamic_indices = [dynamic.index]
       self.kinematic_indices = [kinematic.index]

    def init_constraints(self, scene : Scene, details):
        '''
        Add springs into the static constraints of the scene
        '''
        dynamic = scene.dynamics[self.dynamic_indices[0]]
        kinematic = scene.kinematics[self.kinematic_indices[0]]
        springs = []

        data_x = details.node.flatten('x', dynamic.block_handles)
        data_node_id = details.node.flatten('ID', dynamic.block_handles)

        # Linear search => it will be inefficient for dynamic objects with many nodes
        distance2 = self.distance * self.distance
        for i in range(dynamic.num_nodes()):
            node_pos = data_x[i]
            node_ids = [data_node_id[i]]

            closest_pos = kinematic.get_closest_param(details, node_pos)
            direction = (closest_pos.position - node_pos)
            dist2 = np.inner(direction, direction)
            if dist2 < distance2:
                # add spring
                spring = cpn.AnchorSpring()
                spring.kinematic_index = np.uint32(kinematic.index)
                spring.kinematic_component_index =  np.uint32(closest_pos.index)
                spring.kinematic_component_param = np.float64(closest_pos.t)
                spring.kinematic_component_pos = closest_pos.position
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
       Condition.__init__(self, stiffness, damping, cpn.Spring)
       self.distance = distance
       self.dynamic_indices = [dynamic0.index, dynamic1.index]

    def init_constraints(self, scene : Scene, details):
        '''
        Add springs into the static constraints of the scene
        '''
        springs = []
        dynamic0 = scene.dynamics[self.dynamic_indices[0]]
        dynamic1 = scene.dynamics[self.dynamic_indices[1]]
        distance2 = self.distance * self.distance

        data_x0 = details.node.flatten('x', dynamic0.block_handles)
        data_node_id0 = details.node.flatten('ID', dynamic0.block_handles)
        data_x1 = details.node.flatten('x', dynamic1.block_handles)
        data_node_id1 = details.node.flatten('ID', dynamic1.block_handles)

        for i in range(dynamic0.num_nodes()):
            for j in range(dynamic1.num_nodes()):
                x0 = data_x0[i]
                x1 = data_x1[j]
                direction = (x0 - x1)
                dist2 = np.inner(direction, direction)
                if dist2 < distance2:
                    node_id0 = data_node_id0[i]
                    node_id1 = data_node_id1[j]
                    node_ids = [node_id0, node_id1]

                    # add spring
                    spring = cpn.Spring()
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
       Condition.__init__(self, stiffness, damping, cpn.Spring)
       self.dynamic_indices = [dynamic.index for dynamic in dynamics]

    def init_constraints(self, scene : Scene, details):
        springs = []
        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            for vertex_index in dynamic.edge_ids:
                node_ids = [None, None]
                node_ids[0] = dynamic.get_node_id(vertex_index[0])
                node_ids[1] = dynamic.get_node_id(vertex_index[1])

                # add spring
                spring = cpn.Spring()
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
       Condition.__init__(self, stiffness, damping, cpn.Area)
       self.dynamic_indices = [dynamic.index for dynamic in dynamics]

    def init_constraints(self, scene : Scene, details):
        constraints = []

        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            for vertex_index in dynamic.face_ids:
                node_ids = [None, None, None]
                node_ids[0] = dynamic.get_node_id(vertex_index[0])
                node_ids[1] = dynamic.get_node_id(vertex_index[1])
                node_ids[2] = dynamic.get_node_id(vertex_index[2])

                # add area constraint
                constraint = cpn.Area()
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
       Condition.__init__(self, stiffness, damping, cpn.Bending)
       self.dynamic_indices = [dynamic.index for dynamic in dynamics]

    def init_constraints(self, scene : Scene, details):
        constraints = []

        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            vertex_edges_dict = common.shape.vertex_ids_neighbours(dynamic.edge_ids)
            if self.stiffness > 0.0:
                for vertex_index, vertex_index_neighbour in vertex_edges_dict.items():
                    if (len(vertex_index_neighbour) == 2):
                        node_ids = [None, None, None]
                        node_ids[0] = dynamic.get_node_id(vertex_index_neighbour[0])
                        node_ids[1] = dynamic.get_node_id(vertex_index)
                        node_ids[2] = dynamic.get_node_id(vertex_index_neighbour[1])

                        # add bending constraint
                        constraint = cpn.Bending()
                        constraint.node_IDs = np.copy(node_ids)
                        constraint.stiffness = self.stiffness
                        constraint.damping = self.damping
                        constraints.append(constraint)

        initialize_condition_from_aos(self, constraints, details)
