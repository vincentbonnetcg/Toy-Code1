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
        self.kinematic_animations = [] # animation of kinematic objects
        self.gravity = gravity

    # Data Functions #
    def addDynamic(self, dynamic):
        index = (len(self.dynamics))
        offset = 0
        for i in range(index):
            offset += self.dynamics[i].num_particles

        dynamic.set_indexing(index, offset)
        self.dynamics.append(dynamic)

    def addKinematic(self, kinematic, kinematic_anim = None):
        kinematic.set_indexing(index = (len(self.kinematics)))
        self.kinematics.append(kinematic)
        self.kinematic_animations.append(kinematic_anim)

    def updateKinematics(self, time, dt = 0.0):
        if dt == 0.0:
            return

        for index, kinematic in enumerate(self.kinematics):
            animation = self.kinematic_animations[index]
            if animation:
                position, rotation = animation(time)
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

    # Iterators
    def getConstraintsIterator(self):
        values = []
        for condition in self.conditions:
            values.append(condition.constraints)

        return itertools.chain.from_iterable(values)
