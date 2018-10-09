"""
@author: Vincent Bonnet
@description : a scene contains objects/constraints/kinematics
The scene is a storage for all the data used by the solver
"""

import itertools
import constraints as cn
import numpy as np

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
        dynamic.set_indexing(index, self.computeParticlesOffset(index))
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
            dynamic_constraint_builder.addDynamicConstraints(self)

    def computeParticlesOffset(self, index):
        offset = 0
        for i in range(index):
            offset += self.dynamics[i].num_particles
        return offset

    def numParticles(self):
        numParticles = 0
        for dynamic in self.dynamics:
            numParticles += dynamic.num_particles
        return numParticles

    def attachToKinematic(self, dynamic, kinematic, stiffness, damping, distance):
        # Linear search => it will be inefficient for dynamic objects with many particles
        distance2 = distance * distance
        for particleId, x in enumerate(dynamic.x):
            attachmentPointParams = kinematic.getClosestParametricValues(x)
            attachmentPoint = kinematic.getPointFromParametricValues(attachmentPointParams)
            direction = (attachmentPoint - x)
            dist2 = np.inner(direction, direction)
            if dist2 < distance2:
                constraint = cn.AnchorSpring(stiffness, damping, dynamic, particleId, kinematic, attachmentPointParams)
                self.static_constraints.append(constraint)

    def attachToDynamic(self, dynamic0, dynamic1, stiffness, damping, distance):
        # Linear search => it will be inefficient for dynamic objects with many particles
        distance2 = distance * distance
        for x0i, x0 in enumerate(dynamic0.x):
            for x1i, x1 in enumerate(dynamic1.x):
                direction = (x0 - x1)
                dist2 = np.inner(direction, direction)
                if dist2 < distance2:
                    constraint = cn.Spring(stiffness, damping, [dynamic0, dynamic1], [x0i, x1i])
                    self.static_constraints.append(constraint)

    def add_collision(self, dynamic, kinematic, stiffness, damping):
        collison_builder = cn.KinematicCollisionBuilder(stiffness, damping, dynamic, kinematic)
        self.dynamic_constraint_builders.append(collison_builder)

    def getConstraintsIterator(self):
        values = []
        values.append(self.dynamic_constraints)
        values.append(self.static_constraints)
        for obj in self.dynamics:
            values.append(obj.internal_constraints)

        return itertools.chain.from_iterable(values)
