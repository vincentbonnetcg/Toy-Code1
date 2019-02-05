"""
@author: Vincent Bonnet
@description : Server to be executed on another process
"""

# Development under Spyder IDE
# The sys.path is not set at the project level but where the file is execute
# Append the parent of the parent folder to be able to import modules
import os
import sys
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)

# Import
import system
import pickle

from multiprocessing.managers import SyncManager
from multiprocessing import Queue

def create_setup():
    '''
    Create Setup
    '''
    context = system.Context(time = 0.0, frame_dt = 0.1)
    scene = system.create_wire_scene(context)
    solver = system.ImplicitSolver()

    # Serialize and deserialize
    scene_as_bytes = pickle.dumps(scene)
    scene = pickle.loads(scene_as_bytes)

def make_server_manager(port=8080, authkey=None):
    '''
    Return a server manager with get_job_q and get_result_q methods.
    '''
    job_q = Queue()
    result_q = Queue()

    class JobQueueManager(SyncManager):
        pass

    JobQueueManager.register('get_job_q', callable=lambda: job_q)
    JobQueueManager.register('get_result_q', callable=lambda: result_q)

    manager = SyncManager(address=('', port), authkey = authkey)
    manager.start()
    print('Server started at port %s' % port)
    return manager

def make_client_manager(ip="127.0.0.1", port=8080, authkey=None):
    '''
    Create Client Manager to connect to a server
    '''
    class ServerQueueManager(SyncManager):
        pass

    ServerQueueManager.register('get_job_q')
    ServerQueueManager.register('get_result_q')

    manager = SyncManager(address=(ip, port), authkey = authkey)
    manager.connect()
    print('Client connected to %s:%s' % (ip, port))
    return manager

def execute_server():
    server_manager = make_server_manager()
    # TODO
    pass

def execute_client():
    client_manager = make_client_manager()
    # TODO
    pass

if __name__ == '__main__':
    input("Press Enter to exit server...")
    server_manager.shutdown()
