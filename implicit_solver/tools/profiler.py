"""
@author: Vincent Bonnet
@description : Simple profiler to benchmark code
"""

import time

# Profiler singleton
class ProfilerSingleton(object):

    class __Profiler:
        def __init__(self):
            self.logs = []

        def addLog(self, log):
            self.logs.append(log)

        def clearLogs(self):
            self.logs.clear()

        def printLogs(self):
            print("--- Statistics ---")
            for log in self.logs:
                print(log)

    instance = None

    def __new__(cls):
        if not ProfilerSingleton.instance:
            ProfilerSingleton.instance = ProfilerSingleton.__Profiler()
        return ProfilerSingleton.instance

# timeit decorator
def timeit(method):
    def execute(*args, **kwargs):
        profiler = ProfilerSingleton()
        startTime = time.time()
        result = method(*args, **kwargs)
        endTime = time.time()
        computationTime = endTime - startTime
        profiler.addLog('%r %2.2f sec' % \
              (method.__name__, computationTime))
        return result

    return execute
