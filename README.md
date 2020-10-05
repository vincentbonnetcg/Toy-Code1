# Numerical Bric-a-Brac
Collections of numerical techniques implemented with Python and standard computational packages (Numpy, SciPy, Numba, Matplotlib ...)

### . Implicit Solver
**Folder** [implicit_solver](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/implicit_solver)<br>
**Description** Implicit Solver.

Explore code generation of vectorized code for physics solver.

- Time Integrator : Baraff and Witkin's 
- Spring / Area / Bending / Collision Constraint
- Early code generator for vectorized code

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/implicit_solver/img/implicitSolver_cat.gif" alt="Implicit Solver Beam" height="200px">

### . Path Tracing
**Folder** [path_tracing](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/path_tracing)<br>
**Description** Path Tracer with Python and Numba

- Multithreading with Memory pool per-thread 

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/path_tracing/output/montecarlo_pathtracer_preview.jpg" alt="Monte Carlo Path Tracer" height="200px">

### . Neural Network
**Folder** [neural_network](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/neural_network)<br>
**Description** Neural network study

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/neural_network/deep_deformation/img/deepdeformation.gif" alt="Deep_Deformation" height="200px">

### . Multivariable Optimizations
**File** [optimizations](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/optimizations)<br>
**Description** Implementation of Gradient Descent and Newton's methods

- Optimizer : Gradient Descent, Newton-Raphson, Quasi-Newton (BFGS) 
- Line Search : Backtracking 

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/optimizations/img/optimization.png" alt="Gradient Descent" height="200px">

### . Skeletal Subspace Deformation
**Folder** [skinning](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/skinning)<br>
**Description** Skeletal Subspace Deformation

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/linear_blend_skinning.gif" alt="Skeletal Subspace Deformation" height="200px">

### Stencil Codes
**Folder** [stencil_codes](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/stencil_codes)<br>
**Description** Evaluate stencil codes with Numba (CPU/GPU)

- Poisson Solver, Laplace Inpainting
- Conway's game of life
- Convolution matrix

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/stencil_codes/img/numba_laplace_equation.png" alt="Laplace_Numba" height="200px">

### . Inverse Kinematics
**File** [inverseKinematics.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/miscellaneous/inverseKinematics.py)<br>
**Description** Implementation of inverse kinematics using pseudo-inverse of a jacobian matrix
- Assemble Numerical/Analytic Jacobian matrix (using central difference)
- Solve system with Pseudo-Inverse or Damped Least Squares method

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/inverseKinematics.gif" alt="Inverse Kinematics" height="200px">

### . Dimensionality Reduction
**File** [optimalTransformation.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/miscellaneous/optimalTransformation.py)<br>
**Description** Optimal rigid transformation by using Principal Component Analysis (PCA)

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/optimalTransformation.png" alt="PCA" height="200px">

**File** [svd_compression.ipynb](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/notebooks/image_compression_with_svd.ipynb)<br>
**Description** Image compression with singular value decomposition 

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/notebooks/img/svd_compression.png" alt="SVD" height="200px">

### . Markov Chain - Authors names generator
**File** [markovChain.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/markov_chain/markov_chain.py)<br>
**Description** Generate authors names by using Markov Chain
- Use the Collection of Poems from Poetry Foundation to generate probability matrix

### . Interpolation and Regression
**Folder** [interpolation_regression](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/interpolation_regression)<br>
**Description** Interpolation and regression algorithms

- RBF Interpolation
- Polynomial Regression

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/scatteredDataInterpolation_radialBasisFunction.png" alt="RBF" height="200px">

### . Graph Optimization
**File** [graph_optimization](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/graph_optimization)<br>
**Description** Greedy colouring algorithm

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/graph_optimization/img/graphColouring_greedyAlgorithm.png" alt="Colouring" height="200px">

### . 1D Time Integration

**File** [spring1D_integrator.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/miscellaneous/spring1D_integrator.py)<br>
**Description** Single one-dimensional damped spring to evaluate explicit integrators and compare with analytic solution.
Includes Explicit Euler, RK2, RK4, Semi-Implicit Euler (Euler-Cromer), Leap Frog

<img src="https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/spring1D_integrator.png" alt="Explicit Integrators" height="200px">

### .Jupyter Notebooks

**Folder** [notebooks](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/notebooks)<br>

slowly moving some code into Jupyter Notebooks !
