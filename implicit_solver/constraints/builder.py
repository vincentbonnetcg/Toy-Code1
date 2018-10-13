"""
@author: Vincent Bonnet
@description : constraint builders create a list of constraints from a list of objects
An example of constraint builder is the floorCollisionBuilder
"""

from constraints.spring import AnchorSpring
import numpy as np

class Builder:
    '''
    Base of the constraint builder
    '''
    def __init__(self, stiffness, damping, dynamic, kinematic):
        self.stiffness = stiffness
        self.damping = damping
        self.dynamicIndex = dynamic.index
        self.kinematicIndex = kinematic.index

    def addDynamicConstraints(self, scene):
        # Neighbour search structures or other initialization could happen here
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'addConstraints'")

class KinematicCollisionBuilder(Builder):
    '''
    Base of the floor constraint builder
    '''
    def __init__(self, stiffness, damping, dynamic, kinematic):
        Builder.__init__(self, stiffness, damping, dynamic, kinematic)

    def addDynamicConstraints(self, scene):
        dynamic = scene.dynamics[self.dynamicIndex]
        kinematic = scene.kinematics[self.kinematicIndex]
        for particleId, pos in enumerate(dynamic.x):
            if (kinematic.is_inside(pos)):
                attachmentPointParams = kinematic.getClosestParametricValues(pos)               
                kinematicNormal = kinematic.getNormalFromParametricValues(attachmentPointParams)
                if (np.dot(kinematicNormal, dynamic.v[particleId]) < 0.0):
                    constraint = AnchorSpring(self.stiffness, self.damping, dynamic, particleId, kinematic, attachmentPointParams)
                    scene.dynamic_constraints.append(constraint)

