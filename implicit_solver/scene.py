"""
@author: Vincent Bonnet
@description : a scene contains constraints/objects/kinematics/colliders
"""

import constraints as cn
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

    def addAttachment(self, obj, kinematic, stiffness, damping):
        attachmentPoint = kinematic.getClosestPoint(obj.x[0])
        self.constraints.append(cn.AnchorSpringConstraint(stiffness, damping, [0], attachmentPoint, [obj]))
        
    def getConstraintsIterator(self):
        values = []
        values.append(self.constraints)
        for obj in self.objects:
            values.append(obj.constraints)

        return itertools.chain.from_iterable(values)
        