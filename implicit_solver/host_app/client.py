"""
@author: Vincent Bonnet
@description : Run Client to be executed on another process
"""

from multiprocessing.managers import SyncManager
from multiprocessing import Queue

class ServerQueueManager(SyncManager):
    pass

ServerQueueManager.register('get_job_queue')
ServerQueueManager.register('get_result_queue')


class Client:
    '''
    Client to connect and dispatch commands to a Server
    '''
    def __init__(self):
        self._manager = None
        self._job_queue = None
        self._result_queue = None

    def is_connected(self):
        return self._manager is not None

    def connect_to_server(self, ip="127.0.0.1", port=8080, authkey='12345'):
        try:
            self._manager = ServerQueueManager(address=(ip, port), authkey=bytes(authkey,encoding='utf8'))
            self._manager.connect()
            self._job_queue = self._manager.get_job_queue()
            self._result_queue = self._manager.get_result_queue()
            print('Client connected to %s:%s' % (ip, port))
        except Exception as e:
            self._manager = None
            self._job_queue = None
            self._result_queue = None
            print('Exception raised by client : ' + str(e))

    def run(self, command_name, **kwargs):
        if self.is_connected():
            self._job_queue.put((command_name, kwargs))
            result = self._result_queue.get(block=True)
            return result

    def disconnect_from_server(self):
        if self.is_connected():
            self._job_queue.put('close_server')

