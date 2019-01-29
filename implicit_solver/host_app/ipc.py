"""
@author: Vincent Bonnet
@description : Inter-process communication between client and server via a socket
"""
import socket
import sys

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

    def get_context(self):
        return self.context

class Client(BundleIPC):
    '''
    Client can store a bundle and/or get bundle from server
    '''
    def __init__(self, scene = None, solver = None, context = None):
        BundleIPC.__init__(self, scene, solver, context)
        self.socket = None

    def is_connected(self):
        return self.socket is not None

    def connect_to_external_server(self, host = "localhost", port = 5050):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((host, port))
        except:
            self.socket = None
            print("Client IPC : Connection error:", sys.exc_info()[0])
            return False

        print("Client ICP connected to Server ICP on",self.socket.getpeername())

        return True

    def disconnect_from_external_server(self):
        if self.is_connected():
            message = 'exit'
            self.socket.send(message.encode())
            self.socket.close()

    def initialize(self):
        if super().is_defined():
            return super().initialize()

        # TODO : send request to server
        return False

    def step(self):
        if super().is_defined():
            return super().step()

        # TODO : send request to server
        return False

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

    def get_context(self):
        if super().is_defined():
            return self.solver

        # TODO : send request to server
        return None

