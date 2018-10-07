"""
@author: Vincent Bonnet
@description : constraint builders create a list of constraints from a list of objects
An example of constraint builder is the floorCollisionBuilder
"""

class Builder:
    '''
    Base of the constraint builder
    '''
    def __init__(self, object_indices):
        self.object_indices = object_indices

    def prepareBuilder(self, scene):
        # Neighbour search structures or other initialization could happen here
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'addConstraints'")

    def getConstraints(self, scene):
        # Build the list of constraints here
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'addConstraints'")

class FloorCollisionBuilder:
    '''
    Base of the floor constraint builder
    '''
    def prepareBuilder(self, scene):
        # TODO
        pass

    def addConstraints(self, scene):
        # TODO
        pass

