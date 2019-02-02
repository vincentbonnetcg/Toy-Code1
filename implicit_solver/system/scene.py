"""
@author: Vincent Bonnet
@description : a scene contains objects/constraints/kinematics
The scene stores data in SI unit which are used by the solver
"""

import itertools

class Scene:
    def __init__(self):
        self.dynamics = [] # dynamic objects
        self.kinematics = [] # kinematic objects
        self.animators = [] # animators for kinematic objects
        self.conditions = [] # create static or dynamic constraints
        self.forces = []

    # Data Functions #
    def addDynamic(self, dynamic):
        index = (len(self.dynamics))
        offset = 0
        for i in range(index):
            offset += self.dynamics[i].num_particles

        dynamic.set_indexing(index, offset)
        self.dynamics.append(dynamic)

    def addKinematic(self, kinematic, animator = None):
        kinematic.set_indexing(index = (len(self.kinematics)))
        self.kinematics.append(kinematic)
        self.animators.append(animator)

    def updateKinematics(self, time, dt = 0.0):
        for index, kinematic in enumerate(self.kinematics):
            animation = self.animators[index]
            if animation:
                position, rotation = animation.get_value(time)
                kinematic.state.update(position, rotation, dt)

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

    # Force Functions #
    def addForce(self, force):
        self.forces.append(force)

