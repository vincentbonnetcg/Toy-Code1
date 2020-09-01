"""
@author: Vincent Bonnet
@description : a scene contains objects/constraints/kinematics
The scene stores data in SI unit which are used by the solver
"""

class Scene:
    def __init__(self):
        self.dynamics = [] # dynamic objects
        self.kinematics = [] # kinematic objects
        self.animators = [] # animators for kinematic objects
        self.conditions = [] # create static or dynamic constraints
        self.forces = []

    # Data Functions #
    def add_dynamic(self, dynamic):
        self.dynamics.append(dynamic)

    def add_kinematic(self, kinematic, animator = None):
        self.kinematics.append(kinematic)
        self.animators.append(animator)

    def init_kinematics(self, details, context):
        self.update_kinematics(details, context)

    def update_kinematics(self, details, context):
        for index, kinematic in enumerate(self.kinematics):
            animation = self.animators[index]
            if animation:
                animation.update_kinematic(details, kinematic, context)

    def num_nodes(self):
        num_nodes = 0
        for dynamic in self.dynamics:
            num_nodes += dynamic.num_nodes()
        return num_nodes

    # Constraint Functions #
    def add_condition(self, condition):
        self.conditions.append(condition)

    def init_conditions(self, details):
        for condition in self.conditions:
            condition.init_constraints(details)

    def update_conditions(self, details):
        for condition in self.conditions:
            # Only update the dynamic condition
            if condition.is_static() is False:
                condition.update_constraints(details)

    # Force Functions #
    def add_force(self, force):
        self.forces.append(force)
