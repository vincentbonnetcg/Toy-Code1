"""
@author: Vincent Bonnet
@description : a scene contains objects/constraints/kinematics
The scene is a storage for all the data used by the solver
"""

import itertools

class Scene:
    def __init__(self, gravity):
        self.dynamics = [] # dynamic objects
        self.kinematics = [] # kinematic objects
        self.static_constraints = [] # static constraints (like area/bending/spring)
        self.static_constraint_builders = [] # creates static constraints
        self.dynamic_constraints = [] # dynamic constraints (like sliding/collision)
        self.dynamic_constraint_builders = [] # creates dynamic constraints
        self.gravity = gravity

    # Data Functions #
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

    def numParticles(self):
        numParticles = 0
        for dynamic in self.dynamics:
            numParticles += dynamic.num_particles
        return numParticles

    # Constraint Functions #
    def addStaticConstraintBuilder(self, constraint_builder):
        self.static_constraint_builders.append(constraint_builder)

    def updateStaticConstraints(self):
        self.static_constraints.clear()
        for static_constraint_builder in self.static_constraint_builders:
            static_constraint_builder.add_constraints(self)

    def addDynamicConstraintBuilder(self, constraint_builder):
        self.dynamic_constraint_builders.append(constraint_builder)

    def updateDynamicConstraints(self):
        self.dynamic_constraints.clear()
        for dynamic_constraint_builder in self.dynamic_constraint_builders:
            dynamic_constraint_builder.add_constraints(self)

    def getConstraintsIterator(self):
        values = []
        values.append(self.dynamic_constraints)
        values.append(self.static_constraints)
        for obj in self.dynamics:
            values.append(obj.internal_constraints)

        return itertools.chain.from_iterable(values)
