"""
@author: Vincent Bonnet
@description : Subclasses of the Condition class
"""

import lib.objects.components as cn
from lib.objects import Condition
import numpy as np
import lib.common as common
import lib.common.node_accessor as na
from lib.system.scene import Scene

def initialize_condition_from_aos(condition, array_of_struct):
    # initialize datablock
    num_constraints = len(array_of_struct)
    condition.data.initialize(num_constraints)
    condition.total_constraints = num_constraints

    if (num_constraints == 0):
        return

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
        condition.data.copyto(field_name, new_array)


class KinematicCollisionCondition(Condition):
    '''
    Creates collision constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping):
        Condition.__init__(self, stiffness, damping, cn.AnchorSpring)
        self.dynamic_indices = [dynamic.index]
        self.kinematic_indices = [kinematic.index]

    def is_static(self):
        '''
        Returns False because collision constraints are dynamics
        '''
        return False

    def init_constraints(self, scene : Scene):
        '''
        Add zero-length springs into the dynamic constraints of the scene
        '''
        dynamic = scene.dynamics[self.dynamic_indices[0]]
        kinematic = scene.kinematics[self.kinematic_indices[0]]
        springs = []

        data_x = dynamic.data.flatten('x')
        data_v = dynamic.data.flatten('v')
        data_node_id = dynamic.data.flatten('ID')

        for i in range(dynamic.num_nodes()):
            node_pos = data_x[i]
            node_vel = data_v[i]
            node_id = data_node_id[i]

            if (kinematic.is_inside(node_pos)):
                attachment_point_params = kinematic.get_closest_parametric_point(node_pos)
                kinematicNormal = kinematic.get_normal_from_parametric_point(attachment_point_params)
                if (np.dot(kinematicNormal, node_vel) < 0.0):
                    # add spring
                    spring = cn.AnchorSpring()
                    spring.set_object(scene, node_id, kinematic, attachment_point_params)
                    spring.stiffness = self.stiffness
                    spring.damping = self.damping
                    springs.append(spring)

        initialize_condition_from_aos(self, springs)


class KinematicAttachmentCondition(Condition):
    '''
    Creates attachment constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping, distance):
       Condition.__init__(self, stiffness, damping, cn.AnchorSpring)
       self.distance = distance
       self.dynamic_indices = [dynamic.index]
       self.kinematic_indices = [kinematic.index]

    def init_constraints(self, scene : Scene):
        '''
        Add springs into the static constraints of the scene
        '''
        dynamic = scene.dynamics[self.dynamic_indices[0]]
        kinematic = scene.kinematics[self.kinematic_indices[0]]
        springs = []

        data_x = dynamic.data.flatten('x')
        data_node_id = dynamic.data.flatten('ID')

        # Linear search => it will be inefficient for dynamic objects with many nodes
        distance2 = self.distance * self.distance
        for i in range(dynamic.num_nodes()):
            node_pos = data_x[i]
            node_id = data_node_id[i]

            attachment_point_params = kinematic.get_closest_parametric_point(node_pos)
            attachment_point = kinematic.get_position_from_parametric_point(attachment_point_params)
            direction = (attachment_point - node_pos)
            dist2 = np.inner(direction, direction)
            if dist2 < distance2:
                # add spring
                spring = cn.AnchorSpring()
                spring.set_object(scene, node_id, kinematic, attachment_point_params)
                spring.stiffness = self.stiffness
                spring.damping = self.damping
                springs.append(spring)

        initialize_condition_from_aos(self, springs)

class DynamicAttachmentCondition(Condition):
    '''
    Creates attachment constraint between two dynamic objects
    '''
    def __init__(self, dynamic0, dynamic1, stiffness, damping, distance):
       Condition.__init__(self, stiffness, damping, cn.Spring)
       self.distance = distance
       self.dynamic_indices = [dynamic0.index, dynamic1.index]

    def init_constraints(self, scene : Scene):
        '''
        Add springs into the static constraints of the scene
        '''
        springs = []
        dynamic0 = scene.dynamics[self.dynamic_indices[0]]
        dynamic1 = scene.dynamics[self.dynamic_indices[1]]
        distance2 = self.distance * self.distance

        data_x0 = dynamic0.data.flatten('x')
        data_node_id0 = dynamic0.data.flatten('ID')
        data_x1 = dynamic1.data.flatten('x')
        data_node_id1 = dynamic1.data.flatten('ID')

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
                    spring = cn.Spring()
                    spring.set_object(scene, node_ids)
                    spring.stiffness = self.stiffness
                    spring.damping = self.damping
                    springs.append(spring)

        initialize_condition_from_aos(self, springs)


class EdgeCondition(Condition):
    '''
    Creates Spring constraints
    Replaces edges with Spring constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, stiffness, damping, cn.Spring)
       self.dynamic_indices = [dynamic.index for dynamic in dynamics]

    def init_constraints(self, scene : Scene):
        springs = []
        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            for vertex_index in dynamic.edge_ids:
                node_ids = [None, None]
                node_ids[0] = na.get_node_id_from_vertex_id(dynamic, vertex_index[0])
                node_ids[1] = na.get_node_id_from_vertex_id(dynamic, vertex_index[1])

                # add spring
                spring = cn.Spring()
                spring.set_object(scene, node_ids)
                spring.stiffness = self.stiffness
                spring.damping = self.damping
                springs.append(spring)

        initialize_condition_from_aos(self, springs)


class AreaCondition(Condition):
    '''
    Creates Area constraints
    Replaces triangle with Area constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, stiffness, damping, cn.Area)
       self.dynamic_indices = [dynamic.index for dynamic in dynamics]

    def init_constraints(self, scene : Scene):
        constraints = []

        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            for vertex_index in dynamic.face_ids:
                node_ids = [None, None, None]
                node_ids[0] = na.get_node_id_from_vertex_id(dynamic, vertex_index[0])
                node_ids[1] = na.get_node_id_from_vertex_id(dynamic, vertex_index[1])
                node_ids[2] = na.get_node_id_from_vertex_id(dynamic, vertex_index[2])

                # add area constraint
                constraint = cn.Area()
                constraint.set_object(scene, node_ids)
                constraint.stiffness = self.stiffness
                constraint.damping = self.damping
                constraints.append(constraint)

        initialize_condition_from_aos(self, constraints)

class WireBendingCondition(Condition):
    '''
    Creates Wire Bending constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, stiffness, damping, cn.Bending)
       self.dynamic_indices = [dynamic.index for dynamic in dynamics]

    def init_constraints(self, scene : Scene):
        constraints = []

        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            vertex_edges_dict = common.shape.vertex_ids_neighbours(dynamic.edge_ids)
            if self.stiffness > 0.0:
                for vertex_index, vertex_index_neighbour in vertex_edges_dict.items():
                    if (len(vertex_index_neighbour) == 2):
                        node_ids = [None, None, None]
                        node_ids[0] = na.get_node_id_from_vertex_id(dynamic, vertex_index_neighbour[0])
                        node_ids[1] = na.get_node_id_from_vertex_id(dynamic, vertex_index)
                        node_ids[2] = na.get_node_id_from_vertex_id(dynamic, vertex_index_neighbour[1])

                        # add bending constraint
                        constraint = cn.Bending()
                        constraint.set_object(scene, node_ids)
                        constraint.stiffness = self.stiffness
                        constraint.damping = self.damping
                        constraints.append(constraint)

        initialize_condition_from_aos(self, constraints)
