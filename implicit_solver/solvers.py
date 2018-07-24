"""
@author: Vincent Bonnet
@description : symplectic and backward Euler integrators
"""

import numpy as np

'''
 Base Solver
'''
class BaseSolver:
    def __init__(self, gravity):
        self.gravity = gravity
        
    def step(self, data, dt, gravity):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'step'")

'''
 Implicit Step
 Solve : 
     (M - h * df/dv - h^2 * df/dx) * deltaV = h * (fo + h * df/dx * v0)
       A = (M - h^2 * df/dx)
       b = h * (fo + h * df/dx * v0)
     => A * deltaV = b <=> deltaV = A^-1 * b    
     deltaX = (v0 + deltaV) * h
     v = v + deltaV 
     x = x + deltaX
'''
class ImplicitSolver(BaseSolver):
    def __init__(self, gravity):
        BaseSolver.__init__(self, gravity)

    def step(self, data, dt):   
        data.f.fill(0.0)
    
        # Add gravity
        for i in range(data.numParticles):
            data.f[i] += np.multiply(self.gravity, data.m[i])
    
        # Prepare forces and jacobians
        for constraint in data.constraints:
            constraint.computeForces(data)
            constraint.computeJacobians(data)
            constraint.applyForces(data)
    
        # Assemble the system (Ax=b) where x is the change of velocity
        # Assemble A = (M - h * df/dv - h^2 * df/dx)
        A = np.zeros((data.numParticles * 2, data.numParticles * 2))
        for i in range(data.numParticles):
            massMatrix = np.matlib.identity(2) * data.m[i]
            A[i*2:i*2+2,i*2:i*2+2] = massMatrix
        
        dfdxMatrix = np.zeros((data.numParticles * 2, data.numParticles * 2))
        for constraint in data.constraints:
            ids = constraint.ids
            for fi in range(len(constraint.ids)):
                for xj in range(len(constraint.ids)):
                    Jx = constraint.getJacobianDx(data, fi, xj)
                    dfdxMatrix[ids[fi]*2:ids[fi]*2+2,ids[xj]*2:ids[xj]*2+2] -= (Jx * dt * dt)
    
        A += dfdxMatrix
        
        dfdvMatrix = np.zeros((data.numParticles * 2, data.numParticles * 2))
        for constraint in data.constraints:
            ids = constraint.ids
            for fi in range(len(constraint.ids)):
                for vj in range(len(constraint.ids)):
                    Jv = constraint.getJacobianDv(data, fi, vj)
                    dfdvMatrix[ids[fi]*2:ids[fi]*2+2,ids[vj]*2:ids[vj]*2+2] -= (Jv * dt)
        
        A += dfdvMatrix
        
        # Assemble b = h *( f0 + h * df/dx * v0)
        # (f0 * h) + (df/dx * v0 * h * h)
        b = np.zeros((data.numParticles * 2, 1))
        for i in range(data.numParticles):
            b[i*2:i*2+2] += (np.reshape(data.f[i], (2,1)) * dt)
        
        for constraint in data.constraints:
            ids = constraint.ids
            for fi in range(len(constraint.ids)):
                for xi in range(len(constraint.ids)):
                    Jx = constraint.getJacobianDx(data, fi, xi)
                    b[ids[fi]*2:ids[fi]*2+2] += np.reshape(np.matmul(data.v[ids[xi]], Jx), (2,1)) * dt * dt
    
        # Solve the system (Ax=b)
        # TODO - will be replaced with conjugate gradient or similar
        deltaVArray = np.linalg.solve(A, b)
           
        # Advect
        for i in range(data.numParticles):
            deltaV = [float(deltaVArray[i*2]), float(deltaVArray[i*2+1])]
            deltaX = (data.v[i] + deltaV) * dt
            data.v[i] += deltaV
            data.x[i] += deltaX

'''
 Semi Implicit Step
'''
class SemiImplicitSolver(BaseSolver):
    def __init__(self, gravity):
        BaseSolver.__init__(self, gravity)

    def step(self, data, dt):
        data.f.fill(0.0)
        # Add gravity
        for i in range(data.numParticles):
            data.f[i] += np.multiply(self.gravity, data.m[i])
    
        # Compute and add internal forces
        for constraint in data.constraints:
            constraint.computeForces(data)
            constraint.applyForces(data)
    
        # Integrator
        for i in range(data.numParticles):
            data.v[i] += data.f[i] * data.im[i] * dt
            data.x[i] += data.v[i] * dt
