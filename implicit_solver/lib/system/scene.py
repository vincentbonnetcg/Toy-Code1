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
        index = (len(self.dynamics))
        offset = 0
        for i in range(index):
            offset += self.dynamics[i].num_nodes()

        dynamic.set_indexing(index, offset)
        self.dynamics.append(dynamic)

    def add_kinematic(self, kinematic, animator = None):
        kinematic.set_indexing(index = (len(self.kinematics)))
        self.kinematics.append(kinematic)
        self.animators.append(animator)

    def init_kinematics(self, start_time):
        self.update_kinematics(start_time, dt = 0.0)

    def update_kinematics(self, time, dt):
        for index, kinematic in enumerate(self.kinematics):
            animation = self.animators[index]
            if animation:
                position, rotation = animation.get_value(time)
                kinematic.state.update(position, rotation, dt)

    def num_nodes(self):
        num_nodes = 0
        for dynamic in self.dynamics:
            num_nodes += dynamic.num_nodes()
        return num_nodes

    # Constraint Functions #
    def add_condition(self, condition):
        self.conditions.append(condition)

    def init_conditions(self):
        for condition in self.conditions:
            condition.update_constraints(self)

    def update_conditions(self):
        for condition in self.conditions:
            # Only update the dynamic condition
            if condition.is_static() is False:
                condition.update_constraints(self)

    # Force Functions #
    def add_force(self, force):
        self.forces.append(force)
