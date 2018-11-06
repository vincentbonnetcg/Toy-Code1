"""
@author: Vincent Bonnet
@description : constraint builders create a list of constraints from a list of objects
"""

import constraints as cn
import numpy as np
import objects.shapes as shapes

class Builder:
    '''
    Base of the constraint builder
    '''
    def __init__(self, dynamics, kinematics, stiffness, damping):
        self.stiffness = stiffness
        self.damping = damping
        self.dynamicIndices = [dynamic.index for dynamic in dynamics]
        self.kinematicIndices = [kinematic.index for kinematic in kinematics]
        # Metadata
        self.meta_data = {}

    def is_static(self):
        '''
        Returns whether or not the created constraints are dynamic or static
        Dynamic constraints are recreated every substep
        Static constraints are created at the first frame and valid for the whole simulation
        '''
        return True

    def add_constraints(self, scene):
        # Neighbour search structures or other initialization could happen here
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'addConstraints'")

    def append_constraint(self, scene, constraint):
        constraint.meta_data = self.meta_data
        if self.is_static():
            scene.static_constraints.append(constraint)
        else:
            scene.dynamic_constraints.append(constraint)


class KinematicCollisionBuilder(Builder):
    '''
    Creates collision constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping):
        Builder.__init__(self, [dynamic], [kinematic], stiffness, damping)

    def is_static(self):
        '''
        Returns False because collision constraints are dynamics
        '''
        return False

    def add_constraints(self, scene):
        '''
        Add zero-length springs into the dynamic constraints of the scene
        '''
        dynamic = scene.dynamics[self.dynamicIndices[0]]
        kinematic = scene.kinematics[self.kinematicIndices[0]]
        for particleId, pos in enumerate(dynamic.x):
            if (kinematic.is_inside(pos)):
                attachmentPointParams = kinematic.getClosestParametricValues(pos)
                kinematicNormal = kinematic.getNormalFromParametricValues(attachmentPointParams)
                if (np.dot(kinematicNormal, dynamic.v[particleId]) < 0.0):
                    constraint = cn.AnchorSpring(self.stiffness, self.damping, dynamic, particleId, kinematic, attachmentPointParams)
                    self.append_constraint(scene, constraint)

class KinematicAttachmentBuilder(Builder):
    '''
    Creates attachment constraint between one kinematic and one dynamic object
    '''
    def __init__(self, dynamic, kinematic, stiffness, damping, distance):
       Builder.__init__(self, [dynamic], [kinematic], stiffness, damping)
       self.distance = distance

    def add_constraints(self, scene):
        '''
        Add springs into the static constraints of the scene
        '''
        dynamic = scene.dynamics[self.dynamicIndices[0]]
        kinematic = scene.kinematics[self.kinematicIndices[0]]
        # Linear search => it will be inefficient for dynamic objects with many particles
        distance2 = self.distance * self.distance
        for particleId, x in enumerate(dynamic.x):
            attachmentPointParams = kinematic.getClosestParametricValues(x)
            attachmentPoint = kinematic.getPointFromParametricValues(attachmentPointParams)
            direction = (attachmentPoint - x)
            dist2 = np.inner(direction, direction)
            if dist2 < distance2:
                constraint = cn.AnchorSpring(self.stiffness, self.damping, dynamic, particleId, kinematic, attachmentPointParams)
                self.append_constraint(scene, constraint)

class DynamicAttachmentBuilder(Builder):
    '''
    Creates attachment constraint between two dynamic objects
    '''
    def __init__(self, dynamic0, dynamic1, stiffness, damping, distance):
       Builder.__init__(self, [dynamic0, dynamic1], [], stiffness, damping)
       self.distance = distance

    def add_constraints(self, scene):
        '''
        Add springs into the static constraints of the scene
        '''
        dynamic0 = scene.dynamics[self.dynamicIndices[0]]
        dynamic1 = scene.dynamics[self.dynamicIndices[1]]
        distance2 = self.distance * self.distance
        for x0i, x0 in enumerate(dynamic0.x):
            for x1i, x1 in enumerate(dynamic1.x):
                direction = (x0 - x1)
                dist2 = np.inner(direction, direction)
                if dist2 < distance2:
                    constraint = cn.Spring(self.stiffness, self.damping, [dynamic0, dynamic1], [x0i, x1i])
                    self.append_constraint(scene, constraint)

class SpringBuilder(Builder):
    '''
    Creates Spring constraints
    Replaces edges with Spring constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Builder.__init__(self, dynamics, [], stiffness, damping)

    def add_constraints(self, scene):
        for object_index in self.dynamicIndices:
            dynamic = scene.dynamics[object_index]
            for vertex_index in dynamic.edge_ids:
                constraint = cn.Spring(self.stiffness, self.damping,
                                       [dynamic, dynamic],
                                       [vertex_index[0], vertex_index[1]])
                self.append_constraint(scene, constraint)

class AreaBuilder(Builder):
    '''
    Creates Area constraints
    Replaces triangle with Area constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Builder.__init__(self, dynamics, [], stiffness, damping)

    def add_constraints(self, scene):
        for object_index in self.dynamicIndices:
            dynamic = scene.dynamics[object_index]
            for vertex_index in dynamic.face_ids:
                constraint = cn.Area(self.stiffness, self.damping,
                                     [dynamic, dynamic, dynamic],
                                     [vertex_index[0], vertex_index[1], vertex_index[2]])
                self.append_constraint(scene, constraint)

class WireBendingBuilder(Builder):
    '''
    Creates Wire Bending constraints
    '''
    def __init__(self, dynamics, stiffness, damping):
       Builder.__init__(self, dynamics, [], stiffness, damping)

    def add_constraints(self, scene):
        for object_index in self.dynamicIndices:
            dynamic = scene.dynamics[object_index]
            vertex_edges_dict = shapes.vertex_ids_neighbours(dynamic.edge_ids)
            if self.stiffness > 0.0:
                for vertex_id, vertex_id_neighbour in vertex_edges_dict.items():
                    if (len(vertex_id_neighbour) == 2):
                        constraint =(cn.Bending(self.stiffness, self.damping,
                                                [dynamic, dynamic, dynamic],
                                                [vertex_id_neighbour[0], vertex_id, vertex_id_neighbour[1]]))
                        self.append_constraint(scene, constraint)

