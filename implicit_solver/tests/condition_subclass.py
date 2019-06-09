"""
@author: Vincent Bonnet
@description : Subclasses of the Condition class
"""

import constraints as cn
from objects import Condition
import numpy as np
import core

class KinematicCollisionCondition(Condition):
    '''
    Creates collision constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping):
        Condition.__init__(self, [dynamic], [kinematic], stiffness, damping)

    def is_static(self):
        '''
        Returns False because collision constraints are dynamics
        '''
        return False

    def add_constraints(self, scene):
        '''
        Add zero-length springs into the dynamic constraints of the scene
        '''
        dynamic = scene.dynamics[self.dynamic_indices[0]]
        kinematic = scene.kinematics[self.kinematic_indices[0]]
        for node_index, node_pos in enumerate(dynamic.x):
            if (kinematic.is_inside(node_pos)):
                attachment_point_params = kinematic.get_closest_parametric_point(node_pos)
                kinematicNormal = kinematic.get_normal_from_parametric_point(attachment_point_params)
                if (np.dot(kinematicNormal, dynamic.v[node_index]) < 0.0):
                    node_id = scene.node_id(dynamic.index, node_index)
                    constraint = cn.AnchorSpring(scene, self.stiffness, self.damping, node_id, kinematic,
                                                 attachment_point_params)
                    self.constraints.append(constraint)

class KinematicAttachmentCondition(Condition):
    '''
    Creates attachment constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping, distance):
       Condition.__init__(self, [dynamic], [kinematic], stiffness, damping)
       self.distance = distance
       self.initialize(cn.AnchorSpring)

    def add_constraints(self, scene):
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
                node_id = scene.node_id(dynamic.index, node_index)

                # add spring
                spring = self.data.create_empty_element()
                cn.AnchorSpring.init_element(spring, scene, node_id, kinematic, attachment_point_params)
                Condition.init_element(spring, self.stiffness, self.damping, [node_id])
                springs.append(spring)

        self.data.initialize_from_array(springs)

class DynamicAttachmentCondition(Condition):
    '''
    Creates attachment constraint between two dynamic objects
    '''
    def __init__(self, dynamic0, dynamic1, stiffness, damping, distance):
       Condition.__init__(self, [dynamic0, dynamic1], [], stiffness, damping)
       self.distance = distance

    def add_constraints(self, scene):
        '''
        Add springs into the static constraints of the scene
        '''
        dynamic0 = scene.dynamics[self.dynamic_indices[0]]
        dynamic1 = scene.dynamics[self.dynamic_indices[1]]
        distance2 = self.distance * self.distance
        for x0i, x0 in enumerate(dynamic0.x):
            for x1i, x1 in enumerate(dynamic1.x):
                direction = (x0 - x1)
                dist2 = np.inner(direction, direction)
                if dist2 < distance2:
                    node_ids = []
                    node_ids.append(scene.node_id(dynamic0.index, x0i))
                    node_ids.append(scene.node_id(dynamic1.index, x1i))
                    constraint = cn.Spring(scene, self.stiffness, self.damping, node_ids)
                    self.constraints.append(constraint)

class EdgeCondition(Condition):
    '''
    Creates Spring constraints
    Replaces edges with Spring constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, dynamics, [], stiffness, damping)
       self.initialize(cn.Spring)

    def add_constraints(self, scene):
        # Old - use constraint list
        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            for vertex_index in dynamic.edge_ids:
                node_ids = []
                node_ids.append(scene.node_id(object_index, vertex_index[0]))
                node_ids.append(scene.node_id(object_index, vertex_index[1]))
                constraint = cn.Spring(scene, self.stiffness, self.damping, node_ids)
                self.constraints.append(constraint)

        # New - datablock => TODO : use this instead of self.constraints
        num_constraints = len(self.constraints)
        self.data.initialize(num_constraints)
        for index, constraint in enumerate(self.constraints):
            self.data.stiffness[index] = constraint.stiffness
            self.data.damping[index] = constraint.damping
            self.data.f[index] = constraint.f
            self.data.node_ids[index] = constraint.n_ids
            self.data.dfdx[index] = constraint.dfdx
            self.data.dfdv[index] = constraint.dfdv
            self.data.rest_length[index] = constraint.rest_length

class AreaCondition(Condition):
    '''
    Creates Area constraints
    Replaces triangle with Area constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, dynamics, [], stiffness, damping)

    def add_constraints(self, scene):
        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            for vertex_index in dynamic.face_ids:
                node_ids = []
                node_ids.append(scene.node_id(object_index, vertex_index[0]))
                node_ids.append(scene.node_id(object_index, vertex_index[1]))
                node_ids.append(scene.node_id(object_index, vertex_index[2]))
                constraint = cn.Area(scene, self.stiffness, self.damping, node_ids)
                self.constraints.append(constraint)

class WireBendingCondition(Condition):
    '''
    Creates Wire Bending constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Condition.__init__(self, dynamics, [], stiffness, damping)

    def add_constraints(self, scene):
        for object_index in self.dynamic_indices:
            dynamic = scene.dynamics[object_index]
            vertex_edges_dict = core.shape.vertex_ids_neighbours(dynamic.edge_ids)
            if self.stiffness > 0.0:
                for vertex_index, vertex_index_neighbour in vertex_edges_dict.items():
                    if (len(vertex_index_neighbour) == 2):
                        node_ids = []
                        node_ids.append(scene.node_id(object_index, vertex_index_neighbour[0]))
                        node_ids.append(scene.node_id(object_index, vertex_index))
                        node_ids.append(scene.node_id(object_index, vertex_index_neighbour[1]))

                        constraint =(cn.Bending(scene, self.stiffness, self.damping, node_ids))
                        self.constraints.append(constraint)

