"""
@author: Vincent Bonnet
@description : Array of Structures of Arrays (AoSoA)

Single Block Memory Layout (with x, v, b as channels)
|-----------------------------|
| x[block_size](np.float)     |
| v[block_size](np.float)     |
| b[block_size](np.int)       |
|-----------------------------|
|blockInfo_size        (int64)|
|blockInfo_capacity    (int64)|
|blockInfo_active      (bool) |
|-----------------------------|

blockInfo_size is the number of set elements in the Block
blockInfo_capacity is the maximum numbe of element in the Block
blockInfo_active defines whether or not the Block is active

Datablock is a list of Blocks
"""

import numba
import numpy as np
import keyword
import lib.common.jit.block_utils as block_utils

class DataBlock:

    def __init__(self, class_type, block_size = 100):
        # Data
        self.blocks = numba.typed.List()
        # Datatype
        self.dtype_block = None
        # Default values
        self.defaults = () #  heterogeneous tuple storing defaults value
        # Block size
        self.block_size = block_size
        # class has an ID (-1)
        self.ID_field_index = -1
        # Set class
        self.__set_dtype(class_type)
        self.clear()

    def num_blocks(self):
        return len(self.blocks)

    def block(self, block_index):
        # [0] because the self.blocks[block_index] is an array with one element
        return self.blocks[block_index][0]

    def clear(self):
        '''
        Clear the data on the datablock (it doesn't reset the datatype)
        '''
        self.blocks = numba.typed.List()
        # append inactive block
        # it prevents to have empty list which would break the JIT compile to work
        block = np.empty(1, dtype=self.dtype_block)
        block[0]['blockInfo_active'] = False
        block[0]['blockInfo_capacity'] = self.block_size
        block[0]['blockInfo_size'] = 0
        self.blocks.append(block)

    @classmethod
    def __check_before_add(cls, field_names, name):
        '''
        Raise exception if 'name' cannot be added
        '''
        if name in ['blockInfo_size', 'blockInfo_active', 'blockInfo_capacity']:
            raise ValueError("field name " + name + " is reserved ")

        if keyword.iskeyword(name):
            raise ValueError("field name cannot be a keyword: " + name)

        if name in field_names:
            raise ValueError("field name already used : " + name)

    def __set_dtype(self, class_type):
        '''
        Set data type from the class type
        '''
        inst = class_type()

        # Aosoa data type : (x, y, ...) becomes (self.block_size, x, y, ...)
        block_type = {}
        block_type['names'] = []
        block_type['formats'] = []

        default_values = []
        for name, value in inst.__dict__.items():
            DataBlock.__check_before_add(block_type['names'], name)
            block_type['names'].append(name)
            default_values.append(value)

            data_format = None # tuple(data_type, data_shape)
            if np.isscalar(value):
                # The coma in data_shape (self.block_size,) is essential
                # In case field_shape == self.block_size == 1,
                # it guarantees an array will be produced and not a single value
                data_format = (type(value), (self.block_size,))
            else:
                data_type = value.dtype.type
                data_shape = ([self.block_size] + list(value.shape))
                data_format = (data_type, data_shape)

            block_type['formats'].append(data_format)

        self.defaults = tuple(default_values)

        # add block info
        block_type['names'].append('blockInfo_size')
        block_type['names'].append('blockInfo_capacity')
        block_type['names'].append('blockInfo_active')
        block_type['formats'].append(np.int64)
        block_type['formats'].append(np.int64)
        block_type['formats'].append(np.bool)

        # create datatype
        self.dtype_block = np.dtype(block_type, align=True)

        # set the ID fieldindex (if it exists)
        if 'ID' in block_type['names']:
            self.ID_field_index = block_type['names'].index('ID')

    def initialize(self, num_elements):
        '''
        Initialize blocks and return new block ids
        '''
        self.clear()
        return self.append(num_elements, True)

    def append(self, num_elements, reuse_inactive_block = False, set_defaults = True):
        '''
        Return a list of new blocks
        Initialize with default values
        '''
        block_handles = None

        if self.ID_field_index >= 0:
            block_handles = block_utils.append_blocks_with_ID(self.blocks,
                                                      reuse_inactive_block,
                                                      num_elements)
        else:
            block_handles = block_utils.append_blocks(self.blocks,
                                                      reuse_inactive_block,
                                                      num_elements)

        if set_defaults==False:
            return block_handles

        for block_handle in block_handles:
            block_container = self.blocks[block_handle]
            for field_id, default_value in enumerate(self.defaults):
                if field_id == self.ID_field_index:
                    continue

                block_container[0][field_id][:] = default_value

        return block_handles

    def append_empty(self, num_elements, reuse_inactive_block = False):
        '''
        Return a list of uninitialized blocks
        '''
        return self.append(num_elements, reuse_inactive_block, False)

    def __len__(self):
        return len(self.blocks)

    '''
    Vectorize Functions on blocks
    '''
    def __take_with_id(self, block_handles = []):
        for block_handle in block_handles:
            block_container = self.blocks[block_handle]
            block_data = block_container[0]
            if block_data['blockInfo_active']:
                yield block_container

    def __take(self):
        for block_container in self.blocks:
            block_data = block_container[0]
            if block_data['blockInfo_active']:
                yield block_container

    def get_blocks(self, block_handles = None):
        if block_handles is None:
            return self.__take()

        return self.__take_with_id(block_handles)

    def compute_num_elements(self, block_handles = None):
        return block_utils.compute_num_elements(self.blocks, block_handles)

    def copyto(self, field_name, values, block_handles = None):
        num_elements = 0

        for block_container in self.get_blocks(block_handles):
            block_data = block_container[0]
            begin_index = num_elements
            block_n_elements = block_data['blockInfo_size']
            num_elements += block_n_elements
            end_index = num_elements
            np.copyto(block_data[field_name][0:block_n_elements], values[begin_index:end_index])

    def fill(self, field_name, value, block_handles = None):
        for block_container in self.get_blocks(block_handles):
            block_data = block_container[0]
            block_data[field_name].fill(value)

    def get_field_names(self):
        return self.block(0).dtype.names

    def flatten(self, field_name, block_handles = None):
        '''
        Convert block of array into a single array
        '''
        field_id = self.get_field_names().index(field_name)
        first_value = self.block(0)[field_id][0]
        field_type = first_value.dtype.type
        field_shape = first_value.shape
        field_format =(field_type, field_shape)
        num_elements = self.compute_num_elements(block_handles)
        result = np.empty(num_elements, field_format)

        num_elements = 0
        for block_container in self.get_blocks(block_handles):
            block_data = block_container[0]
            begin_index = num_elements
            block_n_elements = block_data['blockInfo_size']
            num_elements += block_n_elements
            end_index = num_elements
            np.copyto(result[begin_index:end_index], block_data[field_id][0:block_n_elements])

        return result

    def set_active(self, active, block_handles = None):
        for block_container in self.get_blocks(block_handles):
            block_data = block_container[0]
            block_data['blockInfo_active'] = active
