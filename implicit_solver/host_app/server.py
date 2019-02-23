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
parentdir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parentdir)


from host_app.dispatcher import CommandDispatcher
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
global_dispatcher = CommandDispatcher();

def execute_server(port=8080, authkey='12345'):
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

        # Run the command and return result
        if (isinstance(job, str) and job == 'close_server'):
            exit_solver = True
            result_queue.put("server_exit")
        elif (isinstance(job, tuple)):
            command_name = job[0]
            kwargs = job[1]
            result = global_dispatcher.run(command_name, **kwargs)
            result_queue.put(result)
        else:
            result_queue.put('Command not recognized')

    return manager

if __name__ == '__main__':
    server_manager = execute_server()
    input("Press Enter to exit server...")
    server_manager.shutdown()
