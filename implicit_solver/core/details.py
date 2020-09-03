"""
@author: Vincent Bonnet
@description : details contains a collection of datablocks
"""

import core
from collections import namedtuple

class Details:
    '''
    Details contains the datablocks
    '''
    def __init__(self, system_types, group_types):
        self.db = {} # dictionnary of datablocks

        # create datablock
        block_size = 100
        for datatype in system_types:
            self.db[datatype.name()] = core.DataBlock(datatype, block_size)

        # add blocks as attributes
        for system_type in system_types:
            self.add_attribute(system_type.name(), system_type)

        # add bundles as attributes
        for name, types in group_types.items():
            self.add_attribute(name, types)

    def add_attribute(self, name, datatype):
        get_blocks = lambda types : [self.db[datatype.name()].blocks
                                for datatype in types]
        get_names = lambda types : [datatype.name() for datatype in types]

        if isinstance(datatype, (list, tuple)):
            typename = name+'BundleType'
            setattr(self, typename, namedtuple(typename, get_names(datatype)))
            setattr(self, name, getattr(self, typename)(*get_blocks(datatype)))
        else:
            setattr(self, name, self.db[datatype.name()].blocks)

    def datablock_from_typename(self, typename):
        return self.db[typename]

