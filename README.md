# Numerical Bric-a-Brac
Numerical Methods with Python, Numpy and Matplotlib

## . PDE - Explicit Integrators
**File** [spring1D_integrator.py](https://github.com/vincentbonnetcg/Toy-Code/blob/master/spring1D_integrator.py)<br>
**Description** Single one-dimensional damped spring to evaluate explicit integrators and compare with analytic solution.
- Explicit Euler
- RK2
- RK4
- Semi-Implicit Euler (also called Euler-Cromer)
- Leap Frog

![Inverse Kinematics](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/spring1D_integrator.png)

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

![Inverse Kinematics](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/optimisation_gradientDescent.png)

## . Local Deformations
**File** [optimalTransformation_covarianceMatrix.py](https://github.com/vincentbonnetcg/Toy-Code/blob/master/optimalTransformation_covarianceMatrix.py)<br>
**Description** Optimal rigid transformation by using Principal Component Analysis (Eigen Decomposition)
- Compute Covariance Matrix from point cloud

![Inverse Kinematics](https://github.com/vincentbonnetcg/Toy-Code/blob/master/img/optimalTransformation_covarianceMatrix.png)
