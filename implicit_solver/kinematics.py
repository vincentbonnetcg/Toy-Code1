"""
@author: Vincent Bonnet
@description : Kinematic objects used to support animated objects
"""

'''
 Base Kinematics
'''
class BaseKinematics:
    def __init__(self, position):
        self.position = position
        
    def move(self, time):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'move'")

    def draw(self):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'draw'")
    
    def closestPoint(self, point):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'closestPoint'")

