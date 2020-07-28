"""
@author: Vincent Bonnet
@description : Symplectic Euler time integrator (placeholder)
"""

import lib.common as cm
from lib.system.time_integrators import TimeIntegrator

class SymplecticEulerIntegrator(TimeIntegrator):
    def __init__(self):
        TimeIntegrator.__init__(self)

    @cm.timeit
    def prepare_system(self, scene, details, dt):
        '''
        # TODO
        # Reset forces
        for dynamic in scene.dynamics:
            dynamic.data.fill('f', 0.0)

        # Apply external forces
        for force in scene.forces:
            force.apply_forces(scene.dynamics)

        # Apply internal forces
        for condition in scene.conditions:
            condition.compute_gradients(scene)

        apply_constraint_forces_to_nodes(details.conditions(), details.node)
        '''

    @cm.timeit
    def assemble_system(self, details, dt):
        pass

    @cm.timeit
    def solve_system(self, details, dt):
        '''
        # TODO
        for dynamic in scene.dynamics:
            for i in range(dynamic.num_nodes()):
                dynamic.v[i] += dynamic.f[i] * dynamic.im[i] * dt
                dynamic.x[i] += dynamic.v[i] * dt
        '''
