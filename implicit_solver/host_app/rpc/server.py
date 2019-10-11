"""
@author: Vincent Bonnet
@description : Run Server to be executed on another process
"""

'''
 Development under Spyder IDE
 The sys.path is not set at the project level but where the file is execute
 Append the parent of the parent folder to be able to import modules
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(parentdir)

import host_app.rpc as rpc
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
 Global Dispatcher
'''
global_dispatcher = rpc.CommandDispatcher();

def execute_server(print_log = True, port=8013, authkey='12345'):
    '''
    Launch Server
    '''
    manager = JobQueueManager(address=('localhost', port), authkey = bytes(authkey,encoding='utf8'))
    manager.start()
    print('Server started at port %s' % port)
    exit_solver = False
    job_queue = manager.get_job_queue()
    result_queue = manager.get_result_queue()

    while not exit_solver:
        # Collect a job
        job = job_queue.get(block=True)

        # Run the command from client.py and return result
        result = None
        log = ""
        if isinstance(job, tuple):
            command_name = job[0]
            client_name = job[1]
            if command_name == 'close_server':
                exit_solver = True
                result = 'server_exit'
            else:
                kwargs = job[2]
                result = global_dispatcher.run(command_name, **kwargs)

            log = "client{%s} runs command{%s}" % (client_name , command_name)

        else:
            log = 'Command not recognized (SyntaxError)'
            result = "SyntaxError"

        # Print Log
        if print_log:
            print(log)

        # Add result to result_queue
        result_queue.put(result)

    return manager

if __name__ == '__main__':
    server_manager = execute_server()
    input("Press Enter to exit server...")
    server_manager.shutdown()
