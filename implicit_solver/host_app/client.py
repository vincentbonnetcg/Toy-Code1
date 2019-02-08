"""
@author: Vincent Bonnet
@description : Run Client to be executed on another process
"""

from multiprocessing.managers import SyncManager
from multiprocessing import Queue

def make_client_manager(ip="127.0.0.1", port=8080, authkey='12345'):
    '''
    Create Client Manager to connect to a server
    '''
    class ServerQueueManager(SyncManager):
        pass
    ServerQueueManager.register('get_job_queue')

    manager = ServerQueueManager(address=(ip, port), authkey=bytes(authkey,encoding='utf8'))
    manager.connect()
    print('Client connected to %s:%s' % (ip, port))
    return manager

def execute_client():
    manager = make_client_manager()
    queue = manager.get_job_queue()
    queue.put("hello")
    print(queue)
    # TODO
    return manager


if __name__ == '__main__':
    client_manager = execute_client()  
    input("Press Enter to exit client...")