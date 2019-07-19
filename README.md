# Numerical Bric-a-Brac
Numerics with Python and scientific computing packages (Numpy, SciPy, Numba, Matplotlib ...)

## . Implicit Soft-body Solver
**Folder** [implicit_solver](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/implicit_solver)<br>
**Description** Implicit Solver.

- Implicit / Semi-Implicit integrators
- Spring / Area / Bending Constraint
- Numerical Differentiations (Numerical Jacobian and Hessian) with high accuracy order
- Kinematic Objects
- Dynamic/Dynamic and Dynamic/Static attachment
- Kinematic Collision
- Server-Client (IPC) to communicate solver with other processes

![Implicit Solver Beam](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/implicitSolver_beam.gif)
![Implicit Solver Wire](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/implicitSolver_wire.gif)

[1] Baraff, David, and Andrew Witkin. "Large steps in cloth simulation." Proceedings of the 25th annual conference on Computer graphics and interactive techniques. ACM, 1998.

[2] Teschner, Matthias, Bruno Heidelberger, Matthias Muller, and Markus Gross. "A versatile and robust model for geometrically complex deformable solids." In Computer Graphics International, 2004. Proceedings, pp. 312-319. IEEE, 2004.

[3] Martin, Sebastian, Bernhard Thomaszewski, Eitan Grinspun, and Markus Gross. "Example-based elastic materials." In ACM Transactions on Graphics (TOG), vol. 30, no. 4, p. 72. ACM, 2011. 

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
**Description** Evaluate stencil codes with Numba

- Laplace's Equation and Laplace Inpainting (CPU and GPU)
- Cellular Automata GPU
- Image Processing GPU

![Laplace_Numba](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/numba_laplace_equation.png)

[1] Balasalle, James, Mario A. Lopez, and Matthew J. Rutherford. "Optimizing memory access patterns for cellular automata on GPUs." In GPU Computing Gems Jade Edition, pp. 67-75. 2011.

[2] Gardner, Martin. "Mathematical games: The fantastic combinations of John Conway’s new solitaire game “life”." Scientific American 223, no. 4 (1970): 120-123.

## . Neural Network
**Folder** [neural_network](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/neural_network)<br>
**Description** Neural Network study with Python and Keras

- Feed-Forward neural network from scratch
- Convolution neural network on MNIST dataset
- Autoencoder and Variational Autoencoder

![Fashion_MNIST](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/neural_network/img/fashion_mnist_autoencoder.png)<br>

[1] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

## . Inverse Kinematics
**File** [inverseKinematics_withJacobian.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/inverseKinematics_withJacobian.py)<br>
**Description** Implementation of inverse kinematics using pseudo-inverse of a jacobian matrix
- Assemble Numerical/Analytic Jacobian matrix (using central difference)
- Solve system with Pseudo-Inverse or Damped Least Squares method

![Inverse Kinematics](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/inverseKinematics_withJacobian.gif)

[1] Buss, Samuel R. "Introduction to inverse kinematics with jacobian transpose, pseudoinverse and damped least squares methods." IEEE Journal of Robotics and Automation 17.1-19 (2004): 16.

## . Local Deformations
**File** [optimalTransformation_covarianceMatrix.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/optimalTransformation_covarianceMatrix.py)<br>
**Description** Optimal rigid transformation by using Principal Component Analysis (Eigen Decomposition)
- Compute Covariance Matrix from point cloud

![PCA](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/optimalTransformation_covarianceMatrix.png)

## . Interpolation and Regression
**File** [scatteredDataInterpolation_radialBasisFunction.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/interpolation_regression/scatteredDataInterpolation_radialBasisFunction.py)<br>
**Description** Interpolation of a 1D function with Radial Basis Functions

![RBF](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/scatteredDataInterpolation_radialBasisFunction.png)

[1] Powell, Michael JD. "Radial basis functions for multivariable interpolation: a review." Algorithms for approximation (1987).

[2] Powell, Michael JD. "The theory of radial basis function approximation in 1990." Advances in numerical analysis (1992): 105-210.

**File** [polynomialRegression.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/interpolation_regression/polynomialRegression.py)<br>
**Description** Polynomial Linear Regression on 1D dataset

![Regression](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/polynomial_regression.png)

## . Signed distance field (placeholder)
**Folder** [signed_distance](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/signed_distance_field.py)<br>
**Description** Solve Eikonal Equation to compute signed distance field from 2D-polygon

![RBF](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/distance_field.png)

## . Local Optimisations
**File** [optimisation_gradientDescent.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/optimisation_gradientDescent.py)<br>
**Description** Implementation of Gradient Descent for multivariable functions
- with/without normalized step

![Gradient Descent](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/optimisation_gradientDescent.png)

## . PDE - Numerical Integration

**File** [spring1D_integrator.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/spring1D_integrator.py)<br>
**Description** Single one-dimensional damped spring to evaluate explicit integrators and compare with analytic solution.
- Explicit Euler
- RK2
- RK4
- Semi-Implicit Euler (also called Euler-Cromer)
- Leap Frog

![Explicit Integrators](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/spring1D_integrator.png)

## . Graph Colouring
**File** [graphColouring_greedyAlgorithm.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/graphColouring_greedyAlgorithm.py)<br>
**Description** Greedy colouring algorithm - useful to parallelize the solving process of Constrained Systems

![COLOURING](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/graphColouring_greedyAlgorithm.png)


## . Numba - Sandbox Test

### Array Operations

**Folder** [performance](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/performance)<br>
**Description** Evaluate simple 1D array operation with Numba

![Array_Numba](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/performance_test_array.png)<br>
