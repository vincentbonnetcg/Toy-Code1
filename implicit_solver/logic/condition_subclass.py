"""
@author: Vincent Bonnet
@description : Subclasses of the Condition class
"""

import lib.objects.components as cn
from lib.objects import Condition
import numpy as np
import lib.common as common
import lib.common.node_accessor as na

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

    def init_constraints(self, scene):
        '''
        Add zero-length springs into the dynamic constraints of the scene
        '''
        dynamic = scene.dynamics[self.dynamic_indices[0]]
        kinematic = scene.kinematics[self.kinematic_indices[0]]
        springs = []

        for node_index, node_pos in enumerate(dynamic.x):
            if (kinematic.is_inside(node_pos)):
                attachment_point_params = kinematic.get_closest_parametric_point(node_pos)
                kinematicNormal = kinematic.get_normal_from_parametric_point(attachment_point_params)
                if (np.dot(kinematicNormal, dynamic.v[node_index]) < 0.0):
                    node_id = na.node_id(scene, dynamic.index, node_index)

                    # add spring
                    spring = cn.AnchorSpring()
                    spring.set_object(scene, node_id, kinematic, attachment_point_params)
                    spring.stiffness = self.stiffness
                    spring.damping = self.damping
                    springs.append(spring)

        self.data.initialize_from_array(springs)


class KinematicAttachmentCondition(Condition):
    '''
    Creates attachment constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping, distance):
       Condition.__init__(self, stiffness, damping, cn.AnchorSpring)
       self.distance = distance
       self.dynamic_indices = [dynamic.index]
       self.kinematic_indices = [kinematic.index]

    def init_constraints(self, scene):
        '''
        Add springs into the static constraints of the scene
        '''
        dynamic = scene.dynamics[self.dynamic_indices[0]]
        kinematic = scene.kinematics[self.kinematic_indices[0]]
        springs = []

        # Linear search => it will be inefficient for dynamic objects with many nodes
        distance2 = self.distance * self.distance
        for node_index, node_pos in enumerate(dynamic.x):
            attachment_point_params = kinematic.get_closest_parametric_point(node_pos)
            attachment_point = kinematic.get_position_from_parametric_point(attachment_point_params)
            direction = (attachment_point - node_pos)
            dist2 = np.inner(direction, direction)
            if dist2 < distance2:
                node_id = na.node_id(scene, dynamic.index, node_index)

                # add spring
                spring = cn.AnchorSpring()
                spring.set_object(scene, node_id, kinematic, attachment_point_params)
                spring.stiffness = self.stiffness
                spring.damping = self.damping
                springs.append(spring)

        self.data.initialize_from_array(springs)

class DynamicAttachmentCondition(Condition):
    '''
    Creates attachment constraint between two dynamic objects
    '''
    def __init__(self, dynamic0, dynamic1, stiffness, damping, distance):
       Condition.__init__(self, stiffness, damping, cn.Spring)
       self.distance = distance
       self.dynamic_indices = [dynamic0.index, dynamic1.index]

    def init_constraints(self, scene):
        '''
        Add springs into the static constraints of the scene
        '''
        springs = []

        dynamic0 = scene.dynamics[self.dynamic_indices[0]]
        dynamic1 = scene.dynamics[self.dynamic_indices[1]]
        distance2 = self.distance * self.distance
        for x0i, x0 in enumerate(dynamic0.x):
            for x1i, x1 in enumerate(dynamic1.x):
                direction = (x0 - x1)
                dist2 = np.inner(direction, direction)
                if dist2 < distance2:
                    node_ids = [0, 0]
                    node_ids[0] = na.node_id(scene, dynamic0.index, x0i)
                    node_ids[1] = na.node_id(scene, dynamic1.index, x1i)

                    # add spring
                    spring = cn.Spring()
                    spring.set_object(scene, node_ids)
                    spring.stiffness = self.stiffness
                    spring.damping = self.damping
                    springs.append(spring)

        self.data.initialize_from_array(springs)


class EdgeCondition(Condition):
    '''
    Creates Spring constraints
    Replaces edges with Spring constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, stiffness, damping, cn.Spring)
       self.dynamic_indices = [dynamic.index for dynamic in dynamics]

    def init_constraints(self, scene):
        springs = []

        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            for vertex_index in dynamic.edge_ids:
                node_ids = [0, 0]
                node_ids[0] = na.node_id(scene, object_index, vertex_index[0])
                node_ids[1] = na.node_id(scene, object_index, vertex_index[1])

                # add spring
                spring = cn.Spring()
                spring.set_object(scene, node_ids)
                spring.stiffness = self.stiffness
                spring.damping = self.damping
                springs.append(spring)

        self.data.initialize_from_array(springs)

class AreaCondition(Condition):
    '''
    Creates Area constraints
    Replaces triangle with Area constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, stiffness, damping, cn.Area)
       self.dynamic_indices = [dynamic.index for dynamic in dynamics]

    def init_constraints(self, scene):
        constraints = []

        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            for vertex_index in dynamic.face_ids:
                node_ids = [0, 0, 0]
                node_ids[0] = na.node_id(scene, object_index, vertex_index[0])
                node_ids[1] = na.node_id(scene, object_index, vertex_index[1])
                node_ids[2] = na.node_id(scene, object_index, vertex_index[2])

                # add area constraint
                constraint = cn.Area()
                constraint.set_object(scene, node_ids)
                constraint.stiffness = self.stiffness
                constraint.damping = self.damping
                constraints.append(constraint)

        self.data.initialize_from_array(constraints)

class WireBendingCondition(Condition):
    '''
    Creates Wire Bending constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, stiffness, damping, cn.Bending)
       self.dynamic_indices = [dynamic.index for dynamic in dynamics]

    def init_constraints(self, scene):
        constraints = []

        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            vertex_edges_dict = common.shape.vertex_ids_neighbours(dynamic.edge_ids)
            if self.stiffness > 0.0:
                for vertex_index, vertex_index_neighbour in vertex_edges_dict.items():
                    if (len(vertex_index_neighbour) == 2):
                        node_ids = [0, 0, 0]
                        node_ids[0] = na.node_id(scene, object_index, vertex_index_neighbour[0])
                        node_ids[1] = na.node_id(scene, object_index, vertex_index)
                        node_ids[2] = na.node_id(scene, object_index, vertex_index_neighbour[1])

                        # add bending constraint
                        constraint = cn.Bending()
                        constraint.set_object(scene, node_ids)
                        constraint.stiffness = self.stiffness
                        constraint.damping = self.damping
                        constraints.append(constraint)

        self.data.initialize_from_array(constraints)
