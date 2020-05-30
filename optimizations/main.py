"""
@author: Vincent Bonnet
@description : multivariable optimizations - gradient descent
"""

import convexFunctions
import nonConvexFunctions
import render
import optimizer

if __name__ == '__main__':
    # Step parameter
    optimizer.NORMALIZED_STEP = False  # Only Gradient Descent
    optimizer.SCALE_STEP = 0.1
    # Termination condition
    optimizer.MAX_ITERATIONS = 200
    optimizer.THRESHOLD = 1e-04

    # Non convex functions
    render.draw1D(nonConvexFunctions.trigonometry1D, optimizer.GradientDescent)
    render.draw1D(nonConvexFunctions.trigonometry1D, optimizer.NewtonRaphson)
    render.draw2D(nonConvexFunctions.trigonometry2D, optimizer.GradientDescent)
    render.draw2D(nonConvexFunctions.trigonometry2D, optimizer.NewtonRaphson)
    
    # Convex functions
    render.draw2D(convexFunctions.BohachevskyN1, optimizer.GradientDescent)
    render.draw2D(convexFunctions.BohachevskyN1, optimizer.NewtonRaphson)
    render.draw2D(convexFunctions.McCormick, optimizer.GradientDescent)
    render.draw2D(convexFunctions.McCormick, optimizer.NewtonRaphson)
    render.draw2D(convexFunctions.Booth, optimizer.GradientDescent)
    render.draw2D(convexFunctions.Booth, optimizer.NewtonRaphson)
