# Numerical Bric-a-Brac
Collections of numerical techniques implemented with Python and standard computational packages (Numpy, SciPy, Numba, Matplotlib ...)

Numerical Bric-a-Brac for neural network => [Neural-Bric-a-Brac](https://github.com/vincentbonnetcg/Neural-Bric-a-Brac)

## . Implicit Solver
**Folder** [implicit_solver](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/implicit_solver)<br>
**Description** Implicit Solver.

Explore code generation of vectorized code for physics solver.

- Time Integrator : Baraff and Witkin's 
- Spring / Area / Bending / Collision Constraint
- Early code generator for vectorized code

![Implicit Solver Beam](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/implicit_solver/img/implicitSolver_cat.gif)

## . Path Tracing
**Folder** [path_tracing](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/path_tracing)<br>
**Description** Path Tracer with Python and Numba

- Multithreading
- Memory pool per-thread 
- Supersampling (Uniform jitter)

![Monte Carlo Path Tracer](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/path_tracing/output/montecarlo_pathtracer_preview.jpg)
![Monte Carlo Path Tracer](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/path_tracing/output/teapot_pathtracer_preview.jpg)

## . Multivariable Optimizations
**File** [optimizations](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/optimizations)<br>
**Description** Implementation of Gradient Descent and Newton's methods

- Optimizer : Gradient Descent, Newton-Raphson, Quasi-Newton (BFGS) 
- Line Search : Backtracking 

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/optimizations/img/optimization.png" alt="Gradient Descent" width="40%" height="40%">

## . Skeletal Subspace Deformation
**Folder** [skinning](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/skinning)<br>
**Description** Skeletal Subspace Deformation

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/linear_blend_skinning.gif" alt="Skeletal Subspace Deformation" width="40%" height="40%">

## Stencil Codes
**Folder** [stencil_codes](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/stencil_codes)<br>
**Description** Evaluate stencil codes with Numba (CPU/GPU)

- Poisson Solver
- Laplace Inpainting
- Conway's game of life
- Convolution matrix

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/stencil_codes/img/numba_laplace_equation.png" alt="Laplace_Numba" width="40%" height="40%">

## . Inverse Kinematics
**File** [inverseKinematics_withJacobian.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/miscellaneous/inverseKinematics_withJacobian.py)<br>
**Description** Implementation of inverse kinematics using pseudo-inverse of a jacobian matrix
- Assemble Numerical/Analytic Jacobian matrix (using central difference)
- Solve system with Pseudo-Inverse or Damped Least Squares method

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/inverseKinematics_withJacobian.gif" alt="Inverse Kinematics" width="40%" height="40%">

## . Dimensionality Reduction
**File** [optimalTransformation_covarianceMatrix.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/miscellaneous/optimalTransformation_covarianceMatrix.py)<br>
**Description** Optimal rigid transformation by using Principal Component Analysis (PCA)
- Covariance matrix from point cloud
- Best oriented bounding box
- Eigen Decomposition and SVD Decomposition

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/optimalTransformation_covarianceMatrix.png" alt="PCA" width="40%" height="40%">

## . Markov Chain - Authors names generator
**File** [markovChain.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/markov_chain/markov_chain.py)<br>
**Description** Generate authors names by using Markov Chain
- Use the Collection of Poems from Poetry Foundation to generate probability matrix

## . Interpolation and Regression
**Folder** [interpolation_regression](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/interpolation_regression)<br>
**Description** Interpolation and regression algorithms

- RBF Interpolation
- Polynomial Regression

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/scatteredDataInterpolation_radialBasisFunction.png" alt="RBF" width="40%" height="40%">

## . Graph Optimization
**File** [graph_optimization](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/graph_optimization)<br>
**Description** Greedy colouring algorithm

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/graph_optimization/img/graphColouring_greedyAlgorithm.png" alt="Colouring" width="40%" height="40%">

## . 1D Time Integration

**File** [spring1D_integrator.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/miscellaneous/spring1D_integrator.py)<br>
**Description** Single one-dimensional damped spring to evaluate explicit integrators and compare with analytic solution.
Includes Explicit Euler, RK2, RK4, Semi-Implicit Euler (Euler-Cromer), Leap Frog

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/spring1D_integrator.png" alt="Explicit Integrators" width="40%" height="40%">

## .Jupyter Notebooks

**Folder** [notebooks](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/notebooks)<br>

slowly moving some code into Jupyter Notebooks !
