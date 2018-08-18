"""
@author: Vincent Bonnet
@description : a scene contains constraints/objects/kinematics/colliders
"""

class Scene:
    def __init__(self, gravity):
        self.objects = [] # dynamic objects
        self.kinematics = [] # kinematic objects
        self.gravity = gravity
        
    def addObject(self, obj):
        objectId = (len(self.objects))
        self.objects.append(obj)
        obj.setGlobalIds(objectId, self.computeParticlesOffset(objectId))       

    def addKinematic(self, obj):
        self.kinematics.append(obj)

    def updateKinematics(self, time):
        for obj in self.kinematics:
            obj.update(time)

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
