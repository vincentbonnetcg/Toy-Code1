"""
@author: Vincent Bonnet
@description : Simple profiler to benchmark code
"""

import time

class Profiler(object):
    '''
    Profiler Singleton
    '''
    class Log:
        def __init__(self, function_name, call_depth):
            self.function_name = function_name
            self.call_depth = call_depth
            self.elapsed_time = 0
            self.start_time = time.time()

        def __str__(self):
            result_log = '%r %2.3f sec' % (self.function_name, self.elapsed_time)
            num_spaces = self.call_depth * 3
            result_log = result_log.rjust(len(result_log) + num_spaces, ' ')
            return result_log

    class __Profiler:
        def __init__(self):
            self.logs = []
            self.call_depth_counter = 0

        def push_log(self, function_name):
            log = Profiler.Log(function_name, self.call_depth_counter)
            self.logs.append(log)
            self.call_depth_counter += 1
            return log

        def pop_log(self, log):
            log.elapsed_time = time.time() - log.start_time
            self.call_depth_counter -= 1

        def clear_logs(self):
            self.logs.clear()
            self.call_depth_counter = 0

        def print_logs(self):
            print("--- Statistics ---")
            for log in self.logs:
                print(log)

    instance = None

    def __new__(cls):
        if not Profiler.instance:
            Profiler.instance = Profiler.__Profiler()
        return Profiler.instance

def timeit(method):
    '''
    timeit decorator
    '''
    def execute(*args, **kwargs):
        profiler = Profiler()
        log = profiler.push_log(method.__name__)
        result = method(*args, **kwargs)
        profiler.pop_log(log)
        return result

    return execute

