"""
@author: Vincent Bonnet
@description : Utilities function to help convert function into Numba Friendly
"""
import lib.common as common

def numba_friendly(method):
    '''
    Decorator from Datablock to Component
    '''
    def execute(*args):
        arg_list = list(args)

        # Replace common.DataBlock arguments into numpy array
        for arg_id , arg in enumerate(arg_list):
            if (isinstance(arg, common.DataBlock)):
                arg_list[arg_id] = arg.data

        result = method(*arg_list)
        return result

    return execute
