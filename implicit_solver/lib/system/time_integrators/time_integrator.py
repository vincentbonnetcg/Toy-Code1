"""
@author: Vincent Bonnet
@description : Base for time integrators
"""

class TimeIntegrator:
    '''
    Base class for time integrator
    '''
    def prepare_system(self, scene, details, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'prepare_system'")

    def assemble_system(self, details, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'assemble_system'")

    def solve_system(self, details, dt):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'solve_system'")
