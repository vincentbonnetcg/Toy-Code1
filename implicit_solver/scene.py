"""
@author: Vincent Bonnet
@description : a scene contains constraints/objects/kinematics/colliders
"""

import constraints as cn
import numpy as np
import itertools

class Scene:
    def __init__(self, gravity):
        self.dynamics = [] # dynamic objects
        self.kinematics = [] # kinematic objects
        self.constraints = [] # constraints
        self.gravity = gravity
        
    def addDynamic(self, dynamic):
        index = (len(self.dynamics))
        self.dynamics.append(dynamic)
        dynamic.setGlobalIds(index, self.computeParticlesOffset(index))       

    def addKinematic(self, kinematic):
        index = (len(self.kinematics))
        self.kinematics.append(kinematic)
        kinematic.setGlobalIds(index)

    def updateKinematics(self, time):
        for kinematic in self.kinematics:
            kinematic.update(time)

    def computeParticlesOffset(self, index):
        offset = 0
        for i in range(index):
            offset += self.dynamics[i].numParticles
        return offset

    def numParticles(self):
        numParticles = 0
        for dynamic in self.dynamics:
            numParticles += dynamic.numParticles
        return numParticles

    def addAttachment(self, dynamic, kinematic, stiffness, damping, distance):
        # Linear search => it will be inefficient for dynamic objects with many particles
        distance2 = distance * distance
        xid = 0
        for x in dynamic.x:
            attachmentPointParams = kinematic.getClosestParametricValues(x)
            attachmentPoint = kinematic.getPointFromParametricValues(attachmentPointParams)
            direction = (attachmentPoint - x)
            dist2 = np.inner(direction, direction)
            if (dist2 < distance2):
                constraint = cn.AnchorSpringConstraint(stiffness, damping, [dynamic], [xid], kinematic, attachmentPointParams)
                constraint.setGlobalIds(dynamic.index, self.computeParticlesOffset(dynamic.index))
                self.constraints.append(constraint)
            xid+=1
        
    def getConstraintsIterator(self):
        values = []
        values.append(self.constraints)
        for obj in self.dynamics:
            values.append(obj.constraints)

        return itertools.chain.from_iterable(values)
        