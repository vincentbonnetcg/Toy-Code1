# Path Tracing
Monte Carlo Path Tracing using Numba and a memorypool for efficient computation

ThreadPoolExecutor and numba.njit(nogil=True) are used to parallelize the algorithms.

A Memory Pool per-thread is allocated to avoid most of the stalls related to memory allocation.
The memory pool is a light pre-alloacted structure storing a ray(origin, direction), hits (normal, tangent, binormal, material, ..), stats (num intersection), final result ... 


![Monte Carlo Path Tracer](https://github.com/vincentbonnetcg/Numerical-Bric-a-Brac/blob/master/path_tracing/output/montecarlo_pathtracer.jpg)

- Max depth : 15
- Samples per pixel : 5000
- Supersampling(Uniform jitter) : 1
- Moller-Trumbore triangle/ray algorithm

## Resources
[1] Pharr, Matt, Wenzel Jakob, and Greg Humphreys. Physically based rendering: From theory to implementation. Morgan Kaufmann, 2016.
(http://www.pbr-book.org/)

[2] Jensen, Henrik Wann. Realistic image synthesis using photon mapping. AK Peters/CRC Press, 2001.


