"""
@author: Vincent Bonnet
@description : Inter-process communication between client and server via a socket
"""

class BundleIPC:
    '''
    Bundle is simply a container with a scene, a solver and a context
    BundleIPC provides common behaviour between client and server
    '''
    def __init__(self, scene = None, solver = None, context = None):
        self.scene = scene
        self.solver = solver
        self.context = context

    def is_defined(self):
        if self.scene and self.solver and self.context:
            return True
        return False

    def initialize(self):
        if self.is_defined():
            self.solver.initialize(self.scene, self.context)
            return True
        return False

    def step(self):
        if self.is_defined():
            self.context.time += self.context.dt
            self.solver.solveStep(self.scene, self.context)
            return True
        return False

    def get_scene(self):
        return self.scene

    def get_solver(self):
        return self.solver

class Client(BundleIPC):
    '''
    Client can store a bundle and/or get bundle from server
    '''
    def __init__(self, scene = None, solver = None, context = None):
        BundleIPC.__init__(self, scene, solver, context)

    def connect_to_external_server(self, host = "127.0.0.1", port = 5050):
        # TODO : connect to a solver from another process
        pass

    def initialize(self):
        if not super().initialize():
            # TODO : send request to server
            pass

    def step(self):
        if not super().step():
            # TODO : send request to server
            pass

    def get_scene(self):
        if super().is_defined():
            return self.scene

        # TODO : send request to server
        return None

    def get_solver(self):
        if super().is_defined():
            return self.solver

        # TODO : send request to server
        return None

class Server(BundleIPC):
    '''
    Server stores the data and solver informations
    '''
    def __init__(self, scene = None, solver = None, context = None):
        BundleIPC.__init__(self, scene, solver, context)

    def create_server(self, host = "127.0.0.1", port = 5050):
        pass

