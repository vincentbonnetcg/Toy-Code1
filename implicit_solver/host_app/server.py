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

import system
import pickle

from multiprocessing.managers import SyncManager
from multiprocessing import Queue

'''
Custom SyncManager and register global job/result queue
'''
class JobQueueManager(SyncManager):
    pass
global_job_queue = Queue()
global_result_queue = Queue()
def function_job_queue():
   return global_job_queue
def function_result_queue():
    return global_result_queue

JobQueueManager.register('get_job_queue', callable=function_job_queue)
JobQueueManager.register('get_result_queue', callable=function_result_queue)

'''
Global setup and test serialization/deserialization
'''
global_solver = system.ImplicitSolver()
global_context = system.Context(time = 0.0, frame_dt = 0.1)
global_scene = system.Scene()
system.init_wire_scene(global_scene, global_context)


scene_as_bytes = pickle.dumps(global_solver) # serialize scene
global_solver = pickle.loads(scene_as_bytes) # deserialize scene


def make_server_manager(port=8080, authkey='12345'):
    '''
    Return a server manager with get_job_q and get_result_q methods.
    '''
    manager = JobQueueManager(address=('localhost', port), authkey = bytes(authkey,encoding='utf8'))
    manager.start()
    print('Server started at port %s' % port)
    return manager

def execute_server():
    '''
    Execute job queue
    '''
    manager = make_server_manager()
    exit_solver = False
    job_queue = manager.get_job_queue()
    job_queue.put(global_scene)
    while not exit_solver:
        job = job_queue.get()
        if (job == "exit_solver"):
            exit_solver = True
        #time.sleep(1)
        print(job)

    return manager

if __name__ == '__main__':
    server_manager = execute_server()  
    input("Press Enter to exit server...")
    server_manager.shutdown()
