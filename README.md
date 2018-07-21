# Numerical Bric-a-Brac
Numerical Methods with Python, Numpy and Matplotlib

## . Partial Differential Equations - Numerical Integration
**Files** [implicit_solver](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/implicit_solver)<br>
**Description** Implicit Solver

![Implicit Solver](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/implicitSolver.gif)

**File** [spring1D_integrator.py](https://github.com/vincentbonnetcg/Toy-Code/blob/master/spring1D_integrator.py)<br>
**Description** Single one-dimensional damped spring to evaluate explicit integrators and compare with analytic solution.
- Explicit Euler
- RK2
- RK4
- Semi-Implicit Euler (also called Euler-Cromer)
- Leap Frog

![Explicit Integrators](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/spring1D_integrator.png)

## . Inverse Kinematics
**File** [inverseKinematics_pseudoInverse.py](https://github.com/vincentbonnetcg/Toy-Code/blob/master/inverseKinematics_pseudoInverse.py)<br>
**Description** Implementation of inverse kinematics using pseudo-inverse of a jacobian matrix
- Numerical Jacobian matrix (using central difference) or
- Analytic Jacobian matrix

![Inverse Kinematics](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/inverseKinematics_pseudoInverse.png)

## . Local Optimisations
**File** [optimisation_gradientDescent.py](https://github.com/vincentbonnetcg/Toy-Code/blob/master/optimisation_gradientDescent.py)<br>
**Description** Implementation of Gradient Descent for multivariable functions
- with/without normalized step

![Gradient Descent](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/optimisation_gradientDescent.png)

## . Local Deformations
**File** [optimalTransformation_covarianceMatrix.py](https://github.com/vincentbonnetcg/Toy-Code/blob/master/optimalTransformation_covarianceMatrix.py)<br>
**Description** Optimal rigid transformation by using Principal Component Analysis (Eigen Decomposition)
- Compute Covariance Matrix from point cloud

![PCA](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/optimalTransformation_covarianceMatrix.png)

## . Interpolation and Regression
**File** [scatteredDataInterpolation_radialBasisFunction.py](https://github.com/vincentbonnetcg/Toy-Code/blob/master/scatteredDataInterpolation_radialBasisFunction.py)<br>
**Description** Interpolation of a 1D function with Radial Basis Functions

![RBF](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/scatteredDataInterpolation_radialBasisFunction.png)

## . Graph Colouring
**File** [graphColouring_greedyAlgorithm.py](https://github.com/vincentbonnetcg/Toy-Code/blob/master/graphColouring_greedyAlgorithm.py)<br>
**Description** Greedy colouring algorithm - useful to parallelize the solving process of Constrained Systems

![RBF](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/graphColouring_greedyAlgorithm.png)
