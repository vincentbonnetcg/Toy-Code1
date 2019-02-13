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
        self._scene = scene
        self._solver = solver
        self._context = context

    def is_defined(self):
        if self._scene and self._solver and self._context:
            return True
        return False

    def run_command(self):
        '''
        Execute functions from system.setup.commands and system.commands
        '''
        # TODO
        pass

    def scene(self):
        return self._scene

    def solver(self):
        return self._solver

    def context(self):
        return self._context

class Client(BundleIPC):
    '''
    Client can store a bundle and/or get bundle from server
    '''
    def __init__(self, scene = None, solver = None, context = None):
        BundleIPC.__init__(self, scene, solver, context)

    def is_connected(self):
        # TODO - NOT IMPLEMENTED
        return False

    def connect_to_external_server(self, host = "localhost", port = 8080):
        # TODO - NOT IMPLEMENTED
        return False

    def disconnect_from_external_server(self):
        if self.is_connected():
            # TODO - NOT IMPLEMENTED
            pass
