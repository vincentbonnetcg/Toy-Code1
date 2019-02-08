"""
@author: Vincent Bonnet
@description : Run Server to be executed on another process
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
import time

from multiprocessing.managers import SyncManager
from multiprocessing import Queue

# https://eli.thegreenplace.net/2012/01/24/distributed-computing-in-python-with-multiprocessing

# Custom SyncManager
class JobQueueManager(SyncManager):
    pass
global_queue = Queue()
def job_queue():
   return global_queue
JobQueueManager.register('get_job_queue', callable=job_queue)


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

def make_server_manager(port=8080, authkey='12345'):
    '''
    Return a server manager with get_job_q and get_result_q methods.
    '''
    manager = JobQueueManager(address=('localhost', port), authkey = bytes(authkey,encoding='utf8'))
    manager.start()
    print('Server started at port %s' % port)
    return manager

def execute_server():
    manager = make_server_manager()
   
    while True:
        queue = manager.get_job_queue()
        time.sleep(1)
        print(queue.get())

    # TODO- loop on the queue !
    return manager

if __name__ == '__main__':
    create_setup()
    server_manager = execute_server()  
    input("Press Enter to exit server...")
    server_manager.shutdown()
