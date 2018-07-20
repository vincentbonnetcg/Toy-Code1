"""
@author: Vincent Bonnet
@description : rod simulation with backward euler integrator
Implicit formulation and Conjugate Gradient (WIP)
"""

import numpy as np

'''
 Implicit Step
 Solve : 
     (M - h^2 * df/dx) * deltaV = h * (fo + h * df/dx * v0)
       A = (M - h^2 * df/dx)
       b = h * (fo + h * df/dx * v0)
     => A * deltaV = b <=> deltaV = A^-1 * b    
     deltaX = (v0 + deltaV) * h
     v = v + deltaV 
     x = x + deltaX
'''
def implicitStep(data, dt, gravity):   
    data.f.fill(0.0)

    # Add gravity
    for i in range(data.numParticles):
        data.f[i] += np.multiply(gravity, data.m[i])

    # Prepare forces and jacobians
    for constraint in data.constraints:
        constraint.computeForces(data)
        constraint.computeJacobians(data)
        constraint.applyForces(data)

    # Assemble the system (Ax=b) where x is the change of velocity
    # Assemble A = (M - h^2 * df/dx)
    A = np.zeros((data.numParticles * 2, data.numParticles * 2))
    for i in range(data.numParticles):
        massMatrix = np.matlib.identity(2) * data.m[i]
        A[i*2:i*2+2,i*2:i*2+2] = massMatrix
    
    dfdxMatrix = np.zeros((data.numParticles * 2, data.numParticles * 2))
    for constraint in data.constraints:
        ids = constraint.ids
        for fi in range(len(constraint.ids)):
            for xj in range(len(constraint.ids)):
                Jx = constraint.getJacobian(data, fi, xj)
                dfdxMatrix[ids[fi]*2:ids[fi]*2+2,ids[xj]*2:ids[xj]*2+2] -= (Jx * dt * dt)
        
    A += dfdxMatrix
    
    # Assemble b = h *( f0 + h * df/dx * v0)
    # (f0 * h) + (df/dx * v0 * h * h)
    b = np.zeros((data.numParticles * 2, 1))
    for i in range(data.numParticles):
        b[i*2:i*2+2] += (np.reshape(data.f[i], (2,1)) * dt)
    
    for constraint in data.constraints:
        ids = constraint.ids
        for xi in range(len(constraint.ids)):
            fi = xi
            Jx = constraint.getJacobian(data, fi, xi)
            b[fi*2:fi*2+2] += np.reshape(np.matmul(data.v[ids[xi]], Jx), (2,1)) * dt * dt

    # Solve the system (Ax=b)
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
def semiImplicitStep(data, dt, gravity):
    data.f.fill(0.0)
    # Add gravity
    for i in range(data.numParticles):
        data.f[i] += np.multiply(gravity, data.m[i])

    # Compute and add internal forces
    for constraint in data.constraints:
        constraint.computeForces(data)
        constraint.applyForces(data)

    # Integrator
    for i in range(data.numParticles):
        data.v[i] += data.f[i] * data.im[i] * dt
        data.x[i] += data.v[i] * dt



