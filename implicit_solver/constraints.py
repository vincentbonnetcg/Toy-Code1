"""
@author: Vincent Bonnet
@description : Constraint models
"""

import numpy as np

'''
 Base Constraint
 Describes a constraint function and its first (gradient) and second (hessian) derivatives
'''
class BaseConstraint:
    def __init__(self, stiffness, damping, ids):
        self.stiffness = stiffness
        self.damping = damping
        self.ids = ids
        self.f = np.zeros((len(ids), 2)) # TODO - no need to store that - should be not be allocated in Baseconstraint
        self.dfdx = np.zeros((len(ids),2,2)) # TODO - no need to store that - should be not be allocated in Baseconstraint

    def applyForces(self, data):
        for i in range(len(self.ids)):
            data.f[self.ids[i]] += self.f[i]

    def computeForces(self, data):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeForces'")

    def computeJacobians(self, data):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeJacobians'")

    def getJacobian(self, data, fi, xj):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'getJacobian'")

'''
 Anchor Spring Constraint
 Describes a constraint between particle and static point
'''
class AnchorSpringConstraint(BaseConstraint):
    def __init__(self, stiffness, damping, ids, targetPos, data):
       BaseConstraint.__init__(self, stiffness, damping, ids)
       self.restLength = np.linalg.norm(targetPos - data.x[self.ids[0]])
       self.targetPos = targetPos

    def computeForces(self, data):
        x = data.x[self.ids[0]]
        v = data.v[self.ids[0]]
        force = springStretchForce(x, self.targetPos, self.restLength, self.stiffness)
        force += springDampingForce(x, self.targetPos, v, (0,0), self.damping)
        self.f[0] = force * -1
    
    def computeJacobians(self, data):
        x = data.x[self.ids[0]]
        self.dfdx[0] = stretchNumericalJacobiandf0dx0(x,  self.targetPos, self.restLength, self.stiffness) * -1
        
    def getJacobian(self, data, fi, xj):
        return self.dfdx[0]

'''
 Spring Constraint
 Describes a constraint between two particles
'''
class SpringConstraint(BaseConstraint):
    def __init__(self, stiffness, damping, ids, data):
        BaseConstraint.__init__(self, stiffness, damping, ids)
        self.restLength = np.linalg.norm(data.x[ids[0]] - data.x[ids[1]])

    def computeForces(self, data):
        x0 = data.x[self.ids[0]]
        x1 = data.x[self.ids[1]]
        v0 = data.v[self.ids[0]]
        v1 = data.v[self.ids[1]]
        force = springStretchForce(x0, x1, self.restLength, self.stiffness)
        force += springDampingForce(x0, x1, v0, v1, self.damping)
        self.f[0] = force * -1
        self.f[1] = force

    def computeJacobians(self, data):
        x0 = data.x[self.ids[0]]
        x1 = data.x[self.ids[1]]
        self.dfdx[1] = stretchNumericalJacobiandf0dx0(x0, x1, self.restLength, self.stiffness)
        self.dfdx[0] = self.dfdx[1] * -1

    def getJacobian(self, data, fi, xj):
        #(df/dx)ji = (df/dx)ij = Jx 
        #(df/dx)ii = (df/dx)jj = -Jx
        if (fi == xj):
            return self.dfdx[0]
        return self.dfdx[1]

'''
 Spring Utility Functions
'''
def springStretchForce(x0, x1, rest, stiffness):
    direction = x0 - x1
    stretch = np.linalg.norm(direction)
    if (not np.isclose(stretch, 0.0)):
         direction /= stretch

    return direction * ((stretch - rest) * stiffness)

def springDampingForce(x0, x1, v0, v1, damping):
    direction = x0 - x1
    stretch = np.linalg.norm(direction)
    if (not np.isclose(stretch, 0.0)):
        direction /= stretch
    relativeVelocity = v0 - v1
    return direction * (np.dot(relativeVelocity, direction) * damping)

'''
 Jacobian Forces Utility Functions
  |dfx/dx   dfx/dy|
  |dfy/dx   dfy/dy|
'''
def stretchNumericalJacobiandf0dx0(x0, x1, rest, stiffness):
    stencilSize = 0.0001 # stencil size for the central difference
    jacobian = np.zeros(shape=(2,2))
    
    # Derivative respective to x
    rx0 = np.add(x0, [stencilSize, 0])
    lx0 = np.add(x0, [-stencilSize, 0])
    rforce0 = springStretchForce(rx0, x1, rest, stiffness)
    lforce0 = springStretchForce(lx0, x1, rest, stiffness)
    gradientX = (rforce0 - lforce0) / (stencilSize * 2.0)
    
    # Derivative respective to y
    bx0 = np.add(x0, [0, -stencilSize])
    tx0 = np.add(x0, [0, stencilSize])
    bforce0 = springStretchForce(bx0, x1, rest, stiffness)
    tforce0 = springStretchForce(tx0, x1, rest, stiffness)
    gradientY = (tforce0 - bforce0) / (stencilSize * 2.0)
    
    # Set jacobian with gradients
    jacobian[0, 0] = gradientX[0]
    jacobian[1, 0] = gradientX[1]
    jacobian[0, 1] = gradientY[0]
    jacobian[1, 1] = gradientY[1]
    
    return jacobian