# Numerical Bric-a-Brac
Numerics with Python and scientific computing packages (Numpy, SciPy, Numba, Matplotlib ...)

## . Partial Differential Equations - Numerical Integration
**Folder** [implicit_solver](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/implicit_solver)<br>
**Description** Implicit Solver.
- Implicit / Semi-Implicit integrators
- Spring / Area / Bending Constraint
- Numerical Differentiations (Numerical Jacobian and Hessian) with high accuracy order
- Kinematic Objects
- Dynamic/Dynamic and Dynamic/Static attachment
- Collision Dynamic/Static

![Implicit Solver Beam](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/implicitSolver_beam.gif)
![Implicit Solver Wire](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/implicitSolver_wire.gif)

[1] Baraff, David, and Andrew Witkin. "Large steps in cloth simulation." Proceedings of the 25th annual conference on Computer graphics and interactive techniques. ACM, 1998.

[2] Teschner, Matthias, Bruno Heidelberger, Matthias Muller, and Markus Gross. "A versatile and robust model for geometrically complex deformable solids." In Computer Graphics International, 2004. Proceedings, pp. 312-319. IEEE, 2004.

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

![Inverse Kinematics](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/inverseKinematics_pseudoInverse.gif)

[1] Buss, Samuel R. "Introduction to inverse kinematics with jacobian transpose, pseudoinverse and damped least squares methods." IEEE Journal of Robotics and Automation 17.1-19 (2004): 16.

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

## . Numba and Cuda
**File** [numba_imageProcessing.py](https://github.com/vincentbonnetcg/Toy-Code/blob/master/numba_imageProcessing.py)<br>
**Description** Evaluate simple image processing and cellular automata on Cuda

![NUMBA](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/numba_imageProcessing.png)<br>
![NUMBA](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/numba_cellularAutomata.gif)

[1] Balasalle, James, Mario A. Lopez, and Matthew J. Rutherford. "Optimizing memory access patterns for cellular automata on GPUs." In GPU Computing Gems Jade Edition, pp. 67-75. 2011.<br>
[2] Gardner, Martin. "Mathematical games: The fantastic combinations of John Conway’s new solitaire game “life”." Scientific American 223, no. 4 (1970): 120-123.

## . Neural Network
**Folder** [neural_network](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/tree/master/neural_network)<br>
**Description** Folder to study Neural Network

## . Graph Colouring
**File** [graphColouring_greedyAlgorithm.py](https://github.com/vincentbonnetcg/Toy-Code/blob/master/graphColouring_greedyAlgorithm.py)<br>
**Description** Greedy colouring algorithm - useful to parallelize the solving process of Constrained Systems

![COLOURING](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/graphColouring_greedyAlgorithm.png)
