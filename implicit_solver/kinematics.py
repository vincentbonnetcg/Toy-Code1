"""
@author: Vincent Bonnet
@description : Kinematic objects used to support animated objects
"""
import numpy as np

'''
 Base Kinematic
'''
class BaseKinematic:
    def __init__(self, position):
        self.localSpaceVertices = [] # vertices describing the polygon - requires at least one point
        self.position = position
        self.rotation = 0.0 # in degrees
        self.animationFunc = None
        self.index = 0 # set after the object is added to the scene - index in the scene.kinematics[]

    def getWorldSpaceVertices(self):
        theta = np.radians(self.rotation)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        result = np.matmul(self.localSpaceVertices.copy(), R)
        result = np.add(result, self.position)
        return result

    def update(self, time):
        if self.animationFunc:
            state = self.animationFunc(time)
            self.position = state[0]
            self.rotation = state[1]

    def getClosestParametricValues(self, point):
        # return edgeId and line parameter (t) to define point on this edge
        worldSpaceVertices = self.getWorldSpaceVertices()
        if len(worldSpaceVertices) == 0:
            return None
        elif len(worldSpaceVertices) == 1:
            return [-1, 0.0] # No edge

        result = [-1, 0.0]
        minDistance = np.finfo(np.float64).max
        numEdges = len(worldSpaceVertices)
        for edgeId in range(numEdges):
            # project point on line
            edge = worldSpaceVertices[(edgeId+1)%numEdges] - worldSpaceVertices[edgeId]
            d = point - worldSpaceVertices[edgeId]
            u2 = np.inner(edge, edge)
            t = np.dot(d, edge) / u2
            t = max(min(t, 1.0), 0.0)
            projectedPoint = worldSpaceVertices[edgeId] + (t * edge)
            normal = (point - projectedPoint)
            dist2 = np.inner(normal, normal)
            if dist2 < minDistance:
                minDistance = dist2
                result = [edgeId, t]

        return result

    def getPointFromParametricValues(self, parametricValues):
        worldSpaceVertices = self.getWorldSpaceVertices()
        numEdges = len(worldSpaceVertices)
        edgeId = parametricValues[0]
        if edgeId == -1: # a single point
            return worldSpaceVertices[0]
        else:
            A = worldSpaceVertices[edgeId]
            B = worldSpaceVertices[(edgeId+1)%numEdges]
            t = parametricValues[1]
            return A * (1.0 - t) + B * t

        return None

    def getClosestPoint(self, point):
        params = self.getClosestParametricValues(point)
        if params is None:
            return None

        return self.getPointFromParametricValues(params)

'''
 Point Kinematic
'''
class PointKinematic(BaseKinematic):
    def __init__(self, point):
        BaseKinematic.__init__(self, [0, 0])
        self.localSpaceVertices.append(point)

'''
 Rectangle Kinematic
'''
class RectangleKinematic(BaseKinematic):
    def __init__(self, minX, minY, maxX, maxY):
        BaseKinematic.__init__(self, [0, 0])
        width = maxX - minX
        height = maxY - minY
        halfWidth = width * 0.5
        halfHeight = height * 0.5
        self.localSpaceVertices.append([-halfWidth, -halfHeight])
        self.localSpaceVertices.append([-halfWidth, halfHeight])
        self.localSpaceVertices.append([halfWidth, halfHeight])
        self.localSpaceVertices.append([halfWidth, -halfHeight])
        self.position = [minX + halfWidth, minY + halfHeight]
