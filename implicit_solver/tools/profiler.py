"""
@author: Vincent Bonnet
@description : Simple profiler to benchmark code
"""

import time

# Profiler singleton
class Profiler(object):

    def create_log(function_name, time, depth):
        return

    class Log:
        def __init__(self, function_name):
            self.function_name = function_name;
            self.time = 0
            self.call_depth = 0

        def __str__(self):
            result_log = '%r %2.3f sec' % (self.function_name, self.time)
            num_spaces = (self.call_depth-1) * 3
            result_log = result_log.rjust(len(result_log) + num_spaces, ' ')
            return result_log

    class __Profiler:
        def __init__(self):
            self.logs = []
            self.depth_counter = 0

        def add_log(self, function_name):
            log = Profiler.Log(function_name)
            self.logs.append(log)
            return log

        def push_log(self, function_name):
            self.depth_counter += 1
            return self.add_log(function_name)

        def pop_log(self):
            self.depth_counter -= 1

        def clear_logs(self):
            self.logs.clear()
            self.depth_counter = 0

        def print_logs(self):
            print("--- Statistics ---")
            for log in self.logs:
                print(log)

    instance = None

    def __new__(cls):
        if not Profiler.instance:
            Profiler.instance = Profiler.__Profiler()
        return Profiler.instance


# timeit decorator
def timeit(method):
    def execute(*args, **kwargs):
        profiler = Profiler()
        log = profiler.push_log(method.__name__)
        start_time = time.time()
        result = method(*args, **kwargs)
        end_time = time.time()
        log.time = end_time - start_time
        log.call_depth = profiler.depth_counter
        profiler.pop_log()
        return result

    return execute

