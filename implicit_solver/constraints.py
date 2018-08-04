"""
@author: Vincent Bonnet
@description : Constraint descriptions for implicit solver 
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
        self.f = np.zeros((len(ids), 2))
        # Precomputed jacobians.
        # TODO - should improve that to have better support of constraint with more than two particles
        self.dfdx = np.zeros((len(ids),2,2))
        self.dfdv = np.zeros((len(ids),2,2))

    def applyForces(self, data):
        for i in range(len(self.ids)):
            data.f[self.ids[i]] += self.f[i]

    def computeForces(self, data):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeForces'")

    def computeJacobians(self, data):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeJacobians'")

    def getJacobianDx(self, data, fi, xj):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'getJacobianDx'")
        
    def getJacobianDv(self, data, fi, xj):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'getJacobianDv'")

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
        self.f[0] = force
    
    def computeJacobians(self, data):
        x = data.x[self.ids[0]]
        v = data.v[self.ids[0]]
        # Numerical jacobians
        #self.dfdx[0] = numericalJacobian(springStretchForce, 0, x, self.targetPos, self.restLength, self.stiffness)
        #self.dfdv[0] = numericalJacobian(springDampingForce, 2, x, self.targetPos, v, (0,0), self.damping)
        # Analytic jacobians
        self.dfdx[0] = springStretchJacobian(x, self.targetPos, self.restLength, self.stiffness)
        self.dfdv[0] = springDampingJacobian(x, self.targetPos, v, (0, 0), self.damping)
        
    def getJacobianDx(self, data, fi, xj):
        return self.dfdx[0]

    def getJacobianDv(self, data, fi, xj):
        return self.dfdv[0]

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
        self.f[0] = force
        self.f[1] = force * -1

    def computeJacobians(self, data):
        x0 = data.x[self.ids[0]]
        x1 = data.x[self.ids[1]]
        v0 = data.v[self.ids[0]]
        v1 = data.v[self.ids[1]]     
        # Numerical jacobians
        #self.dfdx[0] = numericalJacobian(springStretchForce, 0, x0, x1, self.restLength, self.stiffness)
        #self.dfdv[0] = numericalJacobian(springDampingForce, 2, x0, x1, v0, v1, self.damping)
        # Analytic jacobians
        self.dfdx[0] = springStretchJacobian(x0, x1, self.restLength, self.stiffness)
        self.dfdv[0] = springDampingJacobian(x0, x1, v0, v1, self.damping)
        self.dfdx[1] = self.dfdx[0] * -1
        self.dfdv[1] = self.dfdv[1] * -1

    def getJacobianDx(self, data, fi, xj):
        #(df/dx)ji = (df/dx)ij = Jx 
        #(df/dx)ii = (df/dx)jj = -Jx
        if (fi == xj):
            return self.dfdx[0]
        return self.dfdx[1]

    def getJacobianDv(self, data, fi, xj):
        #(df/dv)ji = (df/dv)ij = Jx 
        #(df/dv)ii = (df/dv)jj = -Jx
        if (fi == xj):
            return self.dfdv[0]
        return self.dfdv[1]

'''
 Constraint Utility Functions
'''
# direction = normalized(x0-x1)
# stretch = norm(direction)
# A = outerProduct(direction, direction)
# I = identity matrix
# J =  -stiffness * [(1 - rest / stretch)(I - A) + A]
def springStretchJacobian(x0, x1, rest, stiffness):
    jacobian = np.zeros(shape=(2, 2))
    direction = x0 - x1
    stretch = np.linalg.norm(direction)
    I = np.identity(2)
    if (not np.isclose(stretch, 0.0)):
        direction /= stretch
        A = np.outer(direction, direction)
        jacobian = -1.0 * stiffness * ((1 - (rest / stretch)) * (I - A) + A)
    else:
        jacobian = -1.0 * stiffness * I
        
    return jacobian

def springDampingJacobian(x0, x1, v0, v1, damping):
    jacobian = np.zeros(shape=(2, 2))
    direction = x1 - x0
    stretch = np.linalg.norm(direction)
    if (not np.isclose(stretch, 0.0)):
        direction /= stretch
        A = np.outer(direction, direction)
        jacobian = -1.0 *damping * A
    
    return jacobian

def springStretchForce(x0, x1, rest, stiffness):
    direction = x1 - x0
    stretch = np.linalg.norm(direction)
    if (not np.isclose(stretch, 0.0)):
         direction /= stretch
    return direction * ((stretch - rest) * stiffness)

def elasticPotentialEnergy(x0, x1, rest, stiffness):
    direction = x1 - x0
    displacement = np.linalg.norm(direction) - rest
    return 0.5 * stiffness * (displacement * displacement)

def springDampingForce(x0, x1, v0, v1, damping):
    direction = x1 - x0
    stretch = np.linalg.norm(direction)
    if (not np.isclose(stretch, 0.0)):
        direction /= stretch
    relativeVelocity = v1 - v0
    return direction * (np.dot(relativeVelocity, direction) * damping)


'''
 Numerical differentiation Functions
'''
# Private function to compute the gradient of a function with n arguments
# Used by numericalJacobian function
def _numericalGradient(function, argumentId, stencils, *args):
    stencilSize = 1e-6 # stencil for the central difference - Move to numericalGradient
    argsListR = list(args)
    argsListL = list(args)   
    argsListR[argumentId] = np.add(args[argumentId], stencils)
    argsListL[argumentId] = np.subtract(args[argumentId], stencils)
    valueR = function(*argsListR)
    valueL = function(*argsListL)
    gradient = (valueR - valueL) / (stencilSize * 2.0)
    return gradient
    
# This function returns a jacobian matrix with the following dimension :
# [function codomain dimension, function domain dimension]
# 'Function codomain dimension' : dimension of the function output
# 'Function domain dimension' : dimension of the input argumentId of the function
# Warning : Only use scalar as argument (no integer/boolean)
def numericalJacobian(function, argumentId, *args):
    stencilSize = 1e-6 # stencil for the central difference - Move to numericalGradient
    functionCodomainDimension = 1
    functionDomainDimension = 1

    gradientList = []

    # compute gradients    
    if (np.isscalar(args[argumentId])):
        functionDomainDimension = 1        
        gradient = _numericalGradient(function, argumentId, stencilSize, *args)        
        gradientList.append(gradient)
    else:
        functionDomainDimension = len(args[argumentId])
        stencils = np.zeros(functionDomainDimension)
        for i in range(functionDomainDimension):
            stencils.fill(0)
            stencils[i] = stencilSize
            gradient = _numericalGradient(function, argumentId, stencils, *args)           
            gradientList.append(gradient)

    # assemble jacobian from gradients
    if len(gradientList)>0:
        if (np.isscalar(gradientList[0])):
            functionCodomainDimension = 1
        else:
            functionCodomainDimension = len(gradientList[0])

        jacobian = np.zeros(shape=(functionCodomainDimension, functionDomainDimension))
        for gradientId in range(len(gradientList)):
            jacobian[0:functionCodomainDimension, gradientId] = gradientList[gradientId]
    
        return jacobian            
