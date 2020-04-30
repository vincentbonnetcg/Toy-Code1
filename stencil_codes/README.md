# Numerical Bric-a-Brac - Stencil Codes

Stencil Codes written with Python and Numba

## Laplace's Equation and Image Restoration (Laplace Inpainting)
**File** [numba_laplace_equation.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/stencil_codes/numba_laplace_equation.py)<br>
**Description** Numba Jacobi iterative solver

- CPU (multithreaded)
- GPU(with shared memory)

![Laplace_Numba](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/numba_laplace_equation.png)
![Laplace_Numba](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/numba_laplace_inpainting.png)

## Conway's game of life (GPU)
**File** [game_of_life.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/stencil_codes/game_of_life.py)<br>
**Description** Numba image operations

![Automata_Numba](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/numba_cellularAutomata.gif)

## Convolution matrix (GPU)
**File** [convolution_matrix.py](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/stencil_codes/convolution_matrix.py)<br>
**Description** Numba image operations

![ConvolutionMatrix_Numba](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/img/numba_image_processing.png)

## Resources

[1] Balasalle, James, Mario A. Lopez, and Matthew J. Rutherford. "Optimizing memory access patterns for cellular automata on GPUs." In GPU Computing Gems Jade Edition, pp. 67-75. 2011.

[2] Gardner, Martin. "Mathematical games: The fantastic combinations of John Conway’s new solitaire game “life”." Scientific American 223, no. 4 (1970): 120-123.

