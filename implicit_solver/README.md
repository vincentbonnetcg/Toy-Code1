# Numerical Bric-a-Brac - Implicit Solver
Implicit Solver written with Python and Numba

- **lib** : framework and core libraries (scene, solver, base classes of shape, object, condition, ...)

- **logic** : implementation of base classes from lib (shape, object, condition, ...)

- **host_app** : bridge for Maya / Houdini / command line

![Implicit Solver Beam](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/implicit_solver/img/implicitSolver_beam.gif)
![Implicit Solver Wire](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/implicit_solver/img/implicitSolver_wire.gif)

### Optimizations papers

Algorithms usually based on Newton methods (or Quasi-Newton such as L-BFGS)  or conjugate gradient. 

[1] Baraff, David, and Andrew Witkin. "Large steps in cloth simulation." Proceedings of the 25th annual conference on Computer graphics and interactive techniques. ACM, 1998.

[2] Martin, Sebastian, Bernhard Thomaszewski, Eitan Grinspun, and Markus Gross. "Example-based elastic materials." In ACM Transactions on Graphics (TOG), vol. 30, no. 4, p. 72. ACM, 2011. 

[3] Gast, Theodore F., Craig Schroeder, Alexey Stomakhin, Chenfanfu Jiang, and Joseph M. Teran. "Optimization integrator for large time steps." IEEE transactions on visualization and computer graphics 21, no. 10 (2015): 1103-1115.

[4] Wang, Huamin, and Yin Yang. "Descent methods for elastic body simulation on the GPU." ACM Transactions on Graphics (TOG) 35, no. 6 (2016): 212.

[5] Liu, Tiantian, Sofien Bouaziz, and Ladislav Kavan. "Quasi-newton methods for real-time simulation of hyperelastic materials." ACM Transactions on Graphics (TOG) 36, no. 3 (2017): 23.

[6] Bouaziz, Sofien, Sebastian Martin, Tiantian Liu, Ladislav Kavan, and Mark Pauly. "Projective dynamics: fusing constraint projections for fast simulation." ACM Transactions on Graphics (TOG) 33, no. 4 (2014): 1-11.


### Domain-specific language papers

Those DSLs simplify the code and separate the data structures to the algorithms.

[6] Kjolstad, Fredrik, Shoaib Kamil, Jonathan Ragan-Kelley, David IW Levin, Shinjiro Sueda, Desai Chen, Etienne Vouga et al. "Simit: A language for physical simulation." ACM Transactions on Graphics (TOG) 35, no. 2 (2016): 20.

[7] Hu, Yuanming, Tzu-Mao Li, Luke Anderson, Jonathan Ragan-Kelley, and Frédo Durand. "Taichi: a language for high-performance computation on spatially sparse data structures." ACM Transactions on Graphics (TOG) 38, no. 6 (2019): 201.  

### Other papers

[8] Teschner, Matthias, Bruno Heidelberger, Matthias Muller, and Markus Gross. "A versatile and robust model for geometrically complex deformable solids." In Computer Graphics International, 2004. Proceedings, pp. 312-319. IEEE, 2004.

[9] Tournier, Maxime, Matthieu Nesme, Benjamin Gilles, and François Faure. "Stable constrained dynamics." ACM Transactions on Graphics (TOG) 34, no. 4 (2015): 132.

[10] Macklin, Miles, Matthias Müller, and Nuttapong Chentanez. "XPBD: position-based simulation of compliant constrained dynamics." In Proceedings of the 9th International Conference on Motion in Games, pp. 49-54. ACM, 2016.

[11] Gissler, Christoph, Stefan Band, Andreas Peer, Markus Ihmsen, and Matthias Teschner. "Generalized drag force for particle-based simulations." Computers & Graphics 69 (2017): 1-11.

[12] Schroeder, Craig. "Practical course on computing derivatives in code." In ACM SIGGRAPH 2019 Courses, p. 22. ACM, 2019.

[13] Luo, Ran, Weiwei Xu, Tianjia Shao, Hongyi Xu, and Yin Yang. "Accelerated complex-step finite difference for expedient deformable simulation." ACM Transactions on Graphics (TOG) 38, no. 6 (2019): 160.


