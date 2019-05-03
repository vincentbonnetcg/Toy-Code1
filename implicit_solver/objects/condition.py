"""
@author: Vincent Bonnet
@description : conditions create a list of constraints from a list of objects
"""

class Condition:
    '''
    Base of a condition
    '''
    def __init__(self, dynamics, kinematics, stiffness, damping):
        self.stiffness = stiffness
        self.damping = damping
        self.dynamic_indices = [dynamic.index for dynamic in dynamics]
        self.kinematic_indices = [kinematic.index for kinematic in kinematics]
        self.constraints = []
        # Metadata
        self.meta_data = {}

    def is_static(self):
        '''
        Returns whether or not the created constraints are dynamic or static
        Dynamic constraints are recreated every substep
        Static constraints are created at initialisation and valid for the whole simulation
        '''
        return True

    def update_constraints(self, scene):
        self.constraints.clear()
        self.add_constraints(scene)

    def add_constraints(self, scene):
        raise NotImplementedError(type(self).__name__ + " needs to implement the method 'add_constraints'")

