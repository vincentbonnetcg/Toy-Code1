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
        self.f = np.zeros((len(ids), 2))
        # Particle identifications
        self.objectIds = np.zeros(len(ids), dtype=int) # set after the constraint is added to the scene
        self.globalIds = np.copy(ids) # set after the constraint is added to the scene
        self.localIds = np.copy(ids) # local particleIds
        # Precomputed jacobians.
        # TODO - should improve that to have better support of constraint with more than two particles
        self.dfdx = np.zeros((len(ids),2,2))
        self.dfdv = np.zeros((len(ids),2,2))
        
    def setGlobalIds(self, objectId, globalOffset):
        self.objectIds.fill(objectId)
        self.globalIds = np.add(self.localIds, globalOffset)

    def applyForces(self, scene):      
        for i in range(len(self.localIds)):
            dynamic = scene.dynamics[self.objectIds[i]] 
            dynamic.f[self.localIds[i]] += self.f[i]

    def computeForces(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeForces'")

    def computeJacobians(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'computeJacobians'")

    def getJacobianDx(self, fi, xj):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'getJacobianDx'")
        
    def getJacobianDv(self, fi, xj):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'getJacobianDv'")

'''
 Anchor Spring Constraint
 Describes a constraint between particle and static point
'''
class AnchorSpringConstraint(BaseConstraint):
    def __init__(self, stiffness, damping, dynamic, ids, kinematic, pointParams):
       BaseConstraint.__init__(self, stiffness, damping, ids)
       targetPos = kinematic.getPointFromParametricValues(pointParams)
       self.restLength = np.linalg.norm(targetPos - dynamic[0].x[self.localIds[0]])
       self.pointParams = pointParams
       self.kinematic = kinematic

    def computeForces(self, scene):
        dynamic = scene.dynamics[self.objectIds[0]]
        x = dynamic.x[self.localIds[0]]
        v = dynamic.v[self.localIds[0]]
        targetPos = self.kinematic.getPointFromParametricValues(self.pointParams)
        force = springStretchForce(x, targetPos, self.restLength, self.stiffness)
        force += springDampingForce(x, targetPos, v, (0,0), self.damping)
        self.f[0] = force
    
    def computeJacobians(self, scene):
        dynamic = scene.dynamics[self.objectIds[0]]
        x = dynamic.x[self.localIds[0]]
        v = dynamic.v[self.localIds[0]]
        targetPos = self.kinematic.getPointFromParametricValues(self.pointParams)
        # Numerical jacobians
        #self.dfdx[0] = numericalJacobian(springStretchForce, 0, x, targetPos, self.restLength, self.stiffness)
        #self.dfdv[0] = numericalJacobian(springDampingForce, 2, x, targetPos, v, (0,0), self.damping)
        # Analytic jacobians
        self.dfdx[0] = springStretchJacobian(x, targetPos, self.restLength, self.stiffness)
        self.dfdv[0] = springDampingJacobian(x, targetPos, v, (0, 0), self.damping)
        
    def getJacobianDx(self, fi, xj):
        return self.dfdx[0]

    def getJacobianDv(self, fi, xj):
        return self.dfdv[0]

'''
 Spring Constraint
 Describes a constraint between two particles
'''
class SpringConstraint(BaseConstraint):
    def __init__(self, stiffness, damping, dynamic, ids):
        BaseConstraint.__init__(self, stiffness, damping, ids)
        self.restLength = np.linalg.norm(dynamic[0].x[ids[0]] - dynamic[1].x[ids[1]])

    def computeForces(self, scene):
        dynamic0 = scene.dynamics[self.objectIds[0]]
        dynamic1 = scene.dynamics[self.objectIds[1]]
        x0 = dynamic0.x[self.localIds[0]]
        x1 = dynamic1.x[self.localIds[1]]
        v0 = dynamic0.v[self.localIds[0]]
        v1 = dynamic1.v[self.localIds[1]]
        force = springStretchForce(x0, x1, self.restLength, self.stiffness)
        force += springDampingForce(x0, x1, v0, v1, self.damping)
        self.f[0] = force
        self.f[1] = force * -1

    def computeJacobians(self, scene):
        dynamic0 = scene.dynamics[self.objectIds[0]]
        dynamic1 = scene.dynamics[self.objectIds[1]]
        x0 = dynamic0.x[self.localIds[0]]
        x1 = dynamic1.x[self.localIds[1]]
        v0 = dynamic0.v[self.localIds[0]]
        v1 = dynamic1.v[self.localIds[1]]     
        # Numerical jacobians
        #self.dfdx[0] = numericalJacobian(springStretchForce, 0, x0, x1, self.restLength, self.stiffness)
        #self.dfdv[0] = numericalJacobian(springDampingForce, 2, x0, x1, v0, v1, self.damping)
        # Analytic jacobians
        self.dfdx[0] = springStretchJacobian(x0, x1, self.restLength, self.stiffness)
        self.dfdv[0] = springDampingJacobian(x0, x1, v0, v1, self.damping)
        self.dfdx[1] = self.dfdx[0] * -1
        self.dfdv[1] = self.dfdv[1] * -1

    def getJacobianDx(self, fi, xj):
        #(df/dx)ji = (df/dx)ij = Jx 
        #(df/dx)ii = (df/dx)jj = -Jx
        if (fi == xj):
            return self.dfdx[0]
        return self.dfdx[1]

    def getJacobianDv(self, fi, xj):
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
