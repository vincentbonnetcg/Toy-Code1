"""
@author: Vincent Bonnet
@description : a scene contains constraints/objects/kinematics/colliders
"""

import constraints as cn
import numpy as np
import itertools

class Scene:
    def __init__(self, gravity):
        self.objects = [] # dynamic objects
        self.kinematics = [] # kinematic objects
        self.constraints = [] # constraints
        self.gravity = gravity
        
    def addObject(self, obj):
        objectId = (len(self.objects))
        self.objects.append(obj)
        obj.setGlobalIds(objectId, self.computeParticlesOffset(objectId))       

    def addKinematic(self, kinematic):
        self.kinematics.append(kinematic)

    def updateKinematics(self, time):
        for kinematic in self.kinematics:
            kinematic.update(time)

    def computeParticlesOffset(self, objectId):
        offset = 0
        for i in range(objectId):
            offset += self.objects[i].numParticles
        return offset

    def numParticles(self):
        numParticles = 0
        for obj in self.objects:
            numParticles += obj.numParticles
        return numParticles

    def addAttachment(self, obj, kinematic, stiffness, damping, distance):
        # Linear search => it will be inefficient for dynamic objects with many particles
        distance2 = distance * distance
        xid = 0
        for x in obj.x:
            attachmentPoint = kinematic.getClosestPoint(x)
            direction = (attachmentPoint - x)
            dist2 = np.inner(direction, direction)
            if (dist2 < distance2):
                self.constraints.append(cn.AnchorSpringConstraint(stiffness, damping, [xid], attachmentPoint, [obj]))
            xid+=1        
        
    def getConstraintsIterator(self):
        values = []
        values.append(self.constraints)
        for obj in self.objects:
            values.append(obj.constraints)

        return itertools.chain.from_iterable(values)
        