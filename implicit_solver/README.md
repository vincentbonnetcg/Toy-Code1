# Numerical Bric-a-Brac - Implicit Solver
Implicit Solver written with Python and Numba

- **lib** : framework and core libraries (scene, solver, base classes of shape, object, condition, ...)

- **logic** : implementation of base classes from lib (shape, object, condition, ...)

- **host_app** : bridge for Maya / Houdini / command line

![Implicit Solver Beam](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/implicit_solver/img/implicitSolver_beam.gif)
![Implicit Solver Wire](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/implicit_solver/img/implicitSolver_wire.gif)

[1] Baraff, David, and Andrew Witkin. "Large steps in cloth simulation." Proceedings of the 25th annual conference on Computer graphics and interactive techniques. ACM, 1998.

[2] Teschner, Matthias, Bruno Heidelberger, Matthias Muller, and Markus Gross. "A versatile and robust model for geometrically complex deformable solids." In Computer Graphics International, 2004. Proceedings, pp. 312-319. IEEE, 2004.

[3] Martin, Sebastian, Bernhard Thomaszewski, Eitan Grinspun, and Markus Gross. "Example-based elastic materials." In ACM Transactions on Graphics (TOG), vol. 30, no. 4, p. 72. ACM, 2011. 

[4] Gast, Theodore F., Craig Schroeder, Alexey Stomakhin, Chenfanfu Jiang, and Joseph M. Teran. "Optimization integrator for large time steps." IEEE transactions on visualization and computer graphics 21, no. 10 (2015): 1103-1115.

[5] Tournier, Maxime, Matthieu Nesme, Benjamin Gilles, and François Faure. "Stable constrained dynamics." ACM Transactions on Graphics (TOG) 34, no. 4 (2015): 132.

[6] Wang, Huamin, and Yin Yang. "Descent methods for elastic body simulation on the GPU." ACM Transactions on Graphics (TOG) 35, no. 6 (2016): 212.

[7] Macklin, Miles, Matthias Müller, and Nuttapong Chentanez. "XPBD: position-based simulation of compliant constrained dynamics." In Proceedings of the 9th International Conference on Motion in Games, pp. 49-54. ACM, 2016.

[8] Liu, Tiantian, Sofien Bouaziz, and Ladislav Kavan. "Quasi-newton methods for real-time simulation of hyperelastic materials." ACM Transactions on Graphics (TOG) 36, no. 3 (2017): 23.

