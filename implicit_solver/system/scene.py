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
        self.conditions = [] # create static or dynamic constraints
        self.gravity = gravity

    # Data Functions #
    def addDynamic(self, dynamic):
        index = (len(self.dynamics))
        offset = 0
        for i in range(index):
            offset += self.dynamics[i].num_particles

        dynamic.set_indexing(index, offset)
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
    def addCondition(self, condition):
        self.conditions.append(condition)

    def updateConditions(self, static = True):
        for condition in self.conditions:
            if condition.is_static() is static:
                condition.update_constraints(self)

    def getConstraintsIterator(self):
        values = []
        for condition in self.conditions:
            values.append(condition.constraints)

        return itertools.chain.from_iterable(values)
