"""
@author: Vincent Bonnet
@description : Kinematic objects used to support animated objects
"""
import numpy as np

'''
 Base Kinematics
'''
class BaseKinematics:
    def __init__(self, position):
        self.localSpaceVertices = [] # vertices describing the polygon
        self.position = position
        self.rotation = 0.0 # in degrees
        self.animationFunc = None

    def getWorldSpaceVertices(self):
        theta = np.radians(self.rotation)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c,-s), (s, c)))
        result = np.matmul( self.localSpaceVertices.copy(), R)
        result = np.add(result, self.position)
        return result
        
    def update(self, time):
        if (self.animationFunc):
            state = self.animationFunc(time)
            self.position = state[0]
            self.rotation = state[1]

    #def closestPoint(self, point):
    #    raise NotImplementedError(type(self).__name__ + " needs to implement the method 'closestPoint'")

'''
 Base Kinematics
'''
class RectangleKinematics(BaseKinematics):
    def __init__(self, position, width, height):
        BaseKinematics.__init__(self, position)
        halfWidth = width * 0.5
        halfHeight = height * 0.5
        self.localSpaceVertices.append([-halfWidth, -halfHeight])
        self.localSpaceVertices.append([-halfWidth, halfHeight])
        self.localSpaceVertices.append([halfWidth, halfHeight])
        self.localSpaceVertices.append([halfWidth,-halfHeight])