"""
@author: Vincent Bonnet
@description : Kinematic objects used to support animated objects
"""

'''
 Base Kinematics
'''
class BaseKinematics:
    def __init__(self, position):
        self.vertices = [] # vertices describing the 
        self.position = position
        self.positionFunc = None
        
    def update(self, time):
        if (self.positionFunc):
            self.position = self.positionFunc(time)

    #def closestPoint(self, point):
    #    raise NotImplementedError(type(self).__name__ + " needs to implement the method 'closestPoint'")

'''
 Base Kinematics
'''
class RectangleKinematics(BaseKinematics):
    def __init__(self, position):
        BaseKinematics.__init__(self, position)
        self.vertices.append([-1.0,-1.0])
        self.vertices.append([-1.0,1.0])
        self.vertices.append([1.0,1.0])
        self.vertices.append([1.0,-1.0])