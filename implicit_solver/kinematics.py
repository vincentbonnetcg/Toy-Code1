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
        self.localSpaceVertices = [] # vertices describing the polygon - requires at least one point
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

    def getClosestPoint(self, point):
        worldSpaceVertices = self.getWorldSpaceVertices()
        if (len(worldSpaceVertices) == 0):
            return None
        elif (len(worldSpaceVertices) == 1):
            return worldSpaceVertices[0]
       
        minPoint = [0, 0]
        minDistance = np.finfo(np.float64).max
        numEdges = len(worldSpaceVertices)
        for edgeId in range(numEdges):
            # project point on line
            edge = worldSpaceVertices[(edgeId+1)%numEdges] - worldSpaceVertices[edgeId]
            d = point - worldSpaceVertices[edgeId]
            u2 = np.inner(edge, edge)
            t = np.dot(d, edge) / u2
            t = max(min(t,1.0),0.0)
            projectedPoint = worldSpaceVertices[edgeId] + (t * edge)
            normal = (point - projectedPoint)
            dist2 = np.inner(normal, normal)
            if (dist2 < minDistance):
                minDistance = dist2
                minPoint = projectedPoint

        return minPoint

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