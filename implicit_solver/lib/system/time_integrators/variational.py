"""
@author: Vincent Bonnet
@description : Variational time integrator (placeholder)
"""

import core
from lib.system.time_integrators import TimeIntegrator

class VariationalIntegrator(TimeIntegrator):
    def __init__(self):
        TimeIntegrator.__init__(self)

    @core.timeit
    def prepare_system(self, scene, details, dt):
        # TODO
        pass

    @core.timeit
    def assemble_system(self, details, dt):
        # TODO
        pass

    @core.timeit
    def solve_system(self, details, dt):
        # TODO
        pass

