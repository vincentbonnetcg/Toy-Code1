"""
@author: Vincent Bonnet
@description : Kinematic objects used to support animated objects
"""
import numpy as np


class BaseKinematic:
    '''
    Base class to represent kinematic objects
    '''
    def __init__(self, position):
        self.localSpaceVertices = [] # vertices describing the polygon - requires at least one point
        self.position = position
        self.rotation = 0.0 # in degrees
        self.animationFunc = None
        self.index = 0 # set after the object is added to the scene - index in the scene.kinematics[]

    def set_indexing(self, index):
        self.index = index

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
        '''
        Returns a pair [edgeId, line parameter (t)] which define
        the closest point on the polygon
        '''
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

    def getNormalFromParametricValues(self, parametricValues):
        worldSpaceVertices = self.getWorldSpaceVertices()
        numEdges = len(worldSpaceVertices)
        edgeId = parametricValues[0]
        if edgeId >= 0:

            # Compute the normal of the convex hull in local space
            A = self.localSpaceVertices[edgeId]
            B = self.localSpaceVertices[(edgeId+1)%numEdges]
            t = parametricValues[1]
            p = A * (1.0 - t) + B * t
            n = [A[1] - B[1], A[0] - B[0]]
            n /= np.linalg.norm(n)

            dot = n[0] * p[0] + n[1] * p[1]
            if (dot < 0.0):
                n[0] *= -1.0
                n[1] *= -1.0

            # Transform the local space normal to world space normal
            theta = np.radians(self.rotation)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s), (s, c)))
            n = np.matmul(n, R)

            return n

        return [0.0, 0.0]

    def is_inside(self, point):
        '''
        Returns whether or not the point is inside the kinematic
        The base is not implemented yet
        '''
        return False

class Point(BaseKinematic):
    '''
    Single kinematic point
    '''
    def __init__(self, point):
        BaseKinematic.__init__(self, [0.0, 0.0])
        self.localSpaceVertices = np.zeros((1, 2))
        self.localSpaceVertices[0] = point

class Rectangle(BaseKinematic):
    '''
    Kinematic rectangle
    '''
    def __init__(self, minX, minY, maxX, maxY):
        BaseKinematic.__init__(self, [0.0, 0.0])
        # TODO - make sure the max/min are correct
        self.width = maxX - minX
        self.height = maxY - minY
        halfWidth = self.width * 0.5
        halfHeight = self.height * 0.5
        self.localSpaceVertices = np.zeros((4, 2))
        self.localSpaceVertices[0] = [-halfWidth, -halfHeight]
        self.localSpaceVertices[1] = [-halfWidth, halfHeight]
        self.localSpaceVertices[2] = [halfWidth, halfHeight]
        self.localSpaceVertices[3] = [halfWidth, -halfHeight]
        self.position = [minX + halfWidth, minY + halfHeight]

    def is_inside(self, point) :
        '''
        Returns whether or not the point is inside the rectangle
        '''
        theta = np.radians(-self.rotation)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        result = np.matmul(point - self.position, R)
        halfWidth = self.width * 0.5
        halfHeight = self.height * 0.5
        if (result[0] > -halfWidth and result[0] < halfWidth and
            result[1] > -halfHeight and result[1] < halfHeight):
            return True

        return False

