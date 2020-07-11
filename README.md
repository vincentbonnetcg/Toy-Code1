# Numerical Bric-a-Brac
Numerics with Python and scientific computing packages (Numpy, SciPy, Numba, Matplotlib ...)

Numerical Bric-a-Brac for neural network => [Neural-Bric-a-Brac](https://github.com/vincentbonnetcg/Neural-Bric-a-Brac)

## . Implicit Solver
**Folder** [implicit_solver](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/implicit_solver)<br>
**Description** Implicit Solver.

- Time Integrator : Baraff and Witkin's 
- Spring / Area / Bending / Collision Constraint
- Numerical Differentiations
- Server-Client (IPC) to communicate solver with other processes

![Implicit Solver Beam](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/implicit_solver/img/implicitSolver_cat.gif)

[1] Baraff, David, and Andrew Witkin. "Large steps in cloth simulation." Proceedings of the 25th annual conference on Computer graphics and interactive techniques. ACM, 1998.

[2] Teschner, Matthias, Bruno Heidelberger, Matthias Muller, and Markus Gross. "A versatile and robust model for geometrically complex deformable solids." In Computer Graphics International, 2004. Proceedings, pp. 312-319. IEEE, 2004.

## . Path Tracing
**Folder** [path_tracing](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/path_tracing)<br>
**Description** Path Tracer with Python and Numba

- Multithreading
- Memory pool per-thread 
- Supersampling (Uniform jitter)

![Monte Carlo Path Tracer](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/path_tracing/output/montecarlo_pathtracer_preview.jpg)
![Monte Carlo Path Tracer](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/path_tracing/output/teapot_pathtracer_preview.jpg)

[1] Pharr, Matt, Wenzel Jakob, and Greg Humphreys. Physically based rendering: From theory to implementation. Morgan Kaufmann, 2016

## . Multivariable Optimizations
**File** [optimizations](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/optimizations)<br>
**Description** Implementation of Gradient Descent and Newton's methods

- Optimizer : Gradient Descent, Newton-Raphson, Quasi-Newton (BFGS) 
- Line Search : Backtracking 

![Gradient Descent](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/optimizations/img/optimization.png)

[1] Boyd, Stephen, and Lieven Vandenberghe. Convex optimization. Cambridge university press, 2004.

## . Skeletal Subspace Deformation
**Folder** [skinning](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/skinning)<br>
**Description** Skeletal Subspace Deformation

- Linear Blend Skinning [1]
- Dual Quaternion Skinning (WIP)
- Pose‐Space deformation (WIP)

![Skeletal Subspace Deformation](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/linear_blend_skinning.gif)

[1] Magnenat-Thalmann, Nadia, Richard Laperrire, and Daniel Thalmann. "Joint-dependent local deformations for hand animation and object grasping." In In Proceedings on Graphics interface’88. 1988.

[2] Lewis, John P., Matt Cordner, and Nickson Fong. "Pose space deformation: a unified approach to shape interpolation and skeleton-driven deformation." In Proceedings of the 27th annual conference on Computer graphics and interactive techniques, pp. 165-172. ACM Press/Addison-Wesley Publishing Co., 2000.

## Stencil Codes
**Folder** [stencil_codes](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/stencil_codes)<br>
**Description** Evaluate stencil codes with Numba (CPU/GPU)

- Poisson Solver
- Laplace Inpainting
- Conway's game of life
- Convolution matrix

![Laplace_Numba](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/stencil_codes/img/numba_laplace_equation.png)

[1] Balasalle, James, Mario A. Lopez, and Matthew J. Rutherford. "Optimizing memory access patterns for cellular automata on GPUs." In GPU Computing Gems Jade Edition, pp. 67-75. 2011.

[2] Gardner, Martin. "Mathematical games: The fantastic combinations of John Conway’s new solitaire game “life”." Scientific American 223, no. 4 (1970): 120-123.

## . Inverse Kinematics
**File** [inverseKinematics_withJacobian.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/miscellaneous/inverseKinematics_withJacobian.py)<br>
**Description** Implementation of inverse kinematics using pseudo-inverse of a jacobian matrix
- Assemble Numerical/Analytic Jacobian matrix (using central difference)
- Solve system with Pseudo-Inverse or Damped Least Squares method

![Inverse Kinematics](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/inverseKinematics_withJacobian.gif)

[1] Buss, Samuel R. "Introduction to inverse kinematics with jacobian transpose, pseudoinverse and damped least squares methods." IEEE Journal of Robotics and Automation 17.1-19 (2004): 16.

## . Dimensionality reduction (PCA)
**File** [optimalTransformation_covarianceMatrix.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/miscellaneous/optimalTransformation_covarianceMatrix.py)<br>
**Description** Optimal rigid transformation by using Principal Component Analysis (Eigen Decomposition)
- Covariance matrix from point cloud
- Best oriented bounding box

![PCA](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/optimalTransformation_covarianceMatrix.png)

## . Markov Chain - Authors names generator
**File** [markovChain.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/markov_chain/markov_chain.py)<br>
**Description** Generate authors names by using Markov Chain
- Use the Collection of Poems from Poetry Foundation to generate probability matrix

## . Interpolation and Regression
**Folder** [interpolation_regression](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/interpolation_regression)<br>
**Description** Interpolation and regression algorithms

- RBF Interpolation
- Polynomial Regression

![RBF](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/scatteredDataInterpolation_radialBasisFunction.png)

[1] Powell, Michael JD. "Radial basis functions for multivariable interpolation: a review." Algorithms for approximation (1987).

[2] Powell, Michael JD. "The theory of radial basis function approximation in 1990." Advances in numerical analysis (1992): 105-210.

## . Graph Optimization
**File** [graph_optimization](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/graph_optimization)<br>
**Description** Greedy colouring algorithm

![Colouring](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/graph_optimization/img/graphColouring_greedyAlgorithm.png)

[1] Fratarcangeli, Marco, Valentina Tibaldo, and Fabio Pellacini. "Vivace: A practical gauss-seidel method for stable soft body dynamics." ACM Transactions on Graphics (TOG) 35, no. 6 (2016): 1-9.

## . Time Integration

**File** [spring1D_integrator.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/miscellaneous/spring1D_integrator.py)<br>
**Description** Single one-dimensional damped spring to evaluate explicit integrators and compare with analytic solution.
- Explicit Euler
- RK2
- RK4
- Semi-Implicit Euler (also called Euler-Cromer)
- Leap Frog

![Explicit Integrators](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/spring1D_integrator.png)

## . Numba - Sandbox Test

### Array Operations

**Folder** [performance](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/performance)<br>
**Description** Evaluate simple 1D array operation with Numba

![Array_Numba](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/performance_test_array.png)<br>

