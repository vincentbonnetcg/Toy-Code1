"""
@author: Vincent Bonnet
@description : Inter-process communication between client and server via a socket
"""

class Client:
    '''
    Client stores the data and solver informations
    '''
    def __init__(self, scene = None, solver = None, context = None):
        # Socket Informations (TODO - future work)
        self.host = "127.0.0.1"
        self.port = 5050
        # Data and Solver
        self.scene = scene
        self.solver = solver
        self.context = context

    def connect_to_external_server(self):
        # TODO : connect to a solver from another process
        pass

    def is_local(self):
        if self.scene and self.solver and self.context:
            return True

        return False

    def initialize(self):
        if self.is_local():
            self.solver.initialize(self.scene, self.context)
        else:
            # TODO : send request to server
            pass

    def step(self):
        if self.is_local():
            self.context.time += self.context.dt
            self.solver.solveStep(self.scene, self.context)
        else:
            # TODO : send request to server
            pass

    def get_scene(self):
        if self.is_local():
            return self.scene

        # TODO : send request to server
        return None

    def get_solver(self):
        if self.is_local():
            return self.solver

        # TODO : send request to server
        return None

