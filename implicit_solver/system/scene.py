"""
@author: Vincent Bonnet
@description : a scene contains objects/constraints/kinematics
The scene is a storage for all the data used by the solver
"""

import itertools
import constraints as cn

class Scene:
    def __init__(self, gravity):
        self.dynamics = [] # dynamic objects
        self.kinematics = [] # kinematic objects
        self.static_constraints = [] # static constraints (like area/bending/spring)
        self.dynamic_constraints = [] # dynamic constraints (like sliding/collision)
        self.dynamic_constraint_builders = [] # creates dynamic constraints
        self.gravity = gravity

    def addDynamic(self, dynamic):
        index = (len(self.dynamics))
        offset = 0
        for i in range(index):
            offset += self.dynamics[i].num_particles
        
        dynamic.set_indexing(index, offset)
        dynamic.create_internal_constraints()
        self.dynamics.append(dynamic)

    def addKinematic(self, kinematic):
        index = (len(self.kinematics))
        kinematic.set_indexing(index)
        self.kinematics.append(kinematic)

    def updateKinematics(self, time):
        for kinematic in self.kinematics:
            kinematic.update(time)

    def updateDynamicConstraints(self):
        self.dynamic_constraints.clear()
        for dynamic_constraint_builder in self.dynamic_constraint_builders:
            dynamic_constraint_builder.add_constraints(self)

    def numParticles(self):
        numParticles = 0
        for dynamic in self.dynamics:
            numParticles += dynamic.num_particles
        return numParticles

    def attachToKinematic(self, dynamic, kinematic, stiffness, damping, distance):
        attachment_builder = cn.KinematicAttachmentBuilder(dynamic, kinematic, stiffness, damping, distance)
        attachment_builder.add_constraints(self)

    def attachToDynamic(self, dynamic0, dynamic1, stiffness, damping, distance):
        attachment_builder = cn.DynamicAttachmentBuilder(dynamic0, dynamic1, stiffness, damping, distance)
        attachment_builder.add_constraints(self)

    def add_collision(self, dynamic, kinematic, stiffness, damping):
        collison_builder = cn.KinematicCollisionBuilder(dynamic, kinematic, stiffness, damping)
        self.dynamic_constraint_builders.append(collison_builder)

    def getConstraintsIterator(self):
        values = []
        values.append(self.dynamic_constraints)
        values.append(self.static_constraints)
        for obj in self.dynamics:
            values.append(obj.internal_constraints)

        return itertools.chain.from_iterable(values)
