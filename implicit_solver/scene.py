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
        
    def addDynamic(self, obj):
        objectId = (len(self.dynamics))
        self.dynamics.append(obj)
        obj.setGlobalIds(objectId, self.computeParticlesOffset(objectId))       

    def addKinematic(self, kinematic):
        self.kinematics.append(kinematic)

    def updateKinematics(self, time):
        for kinematic in self.kinematics:
            kinematic.update(time)

    def computeParticlesOffset(self, objectId):
        offset = 0
        for i in range(objectId):
            offset += self.dynamics[i].numParticles
        return offset

    def numParticles(self):
        numParticles = 0
        for dynamic in self.dynamics:
            numParticles += dynamic.numParticles
        return numParticles

    def addAttachment(self, obj, kinematic, stiffness, damping, distance):
        # Linear search => it will be inefficient for dynamic objects with many particles
        distance2 = distance * distance
        xid = 0
        for x in obj.x:
            attachmentPointParams = kinematic.getClosestParametricValues(x)
            attachmentPoint = kinematic.getPointFromParametricValues(attachmentPointParams)
            direction = (attachmentPoint - x)
            dist2 = np.inner(direction, direction)
            if (dist2 < distance2):
                self.constraints.append(cn.AnchorSpringConstraint(stiffness, damping, [obj], [xid], kinematic, attachmentPointParams))
            xid+=1
        
    def getConstraintsIterator(self):
        values = []
        values.append(self.constraints)
        for obj in self.dynamics:
            values.append(obj.constraints)

        return itertools.chain.from_iterable(values)
        