"""
@author: Vincent Bonnet
@description : Variational time integrator (placeholder)
"""

import lib.common as cm
from lib.system.time_integrators import TimeIntegrator

class VariationalIntegrator(TimeIntegrator):
    def __init__(self):
        TimeIntegrator.__init__(self)

    @cm.timeit
    def prepare_system(self, scene, details, dt):
        # TODO
        pass

    @cm.timeit
    def assemble_system(self, details, dt):
        # TODO
        pass

    @cm.timeit
    def solve_system(self, details, dt):
        # TODO
        pass

