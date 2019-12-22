"""
@author: Vincent Bonnet
@description : Array of Structures of Arrays (AoSoA)
Example :
data = DataBlock()
data.initialize(10)
print(data.field_a)
print(data.field_b)
"""

import math
import numpy as np
import keyword
import lib.common.jit.node_accessor as na

class DataBlock:

    def __init__(self, class_type, block_size = 100):
        # Data
        self.blocks = list()
        # Data type
        self.dtype_dict = dict()
        self.dtype_dict['names'] = list() # list of names
        self.dtype_dict['formats'] = list() # list of tuples (data_type, data_shape)
        self.dtype_dict['defaults'] = list() # list of default values (should match formats)
        # Block size
        self.block_size = block_size
        # Set class
        self.__set_field_from_type(class_type)

    def num_blocks(self):
        return len(self.blocks)

    def clear(self):
        '''
        Clear the data on the datablock (it doesn't reset the datatype)
        '''
        self.blocks.clear()

    def __check_before_add(self, name):
        '''
        Raise exception if 'name' cannot be added
        '''
        if name in ['blockInfo_numElements']:
            raise ValueError("field name " + name + " is reserved ")

        if keyword.iskeyword(name):
            raise ValueError("field name cannot be a keyword: " + name)

        if name in self.dtype_dict['names']:
            raise ValueError("field name already used : " + name)

    def __set_field_from_type(self, class_type):
        '''
        Add fields
        '''
        data_type = None
        data_shape = None
        inst = class_type()

        for name, value in inst.__dict__.items():
            self.__check_before_add(name)

            if np.isscalar(value):
                data_type = type(value)
                data_shape = 1
            else:
                data_type = value.dtype.type
                data_shape = value.shape

            self.dtype_dict['names'].append(name)
            self.dtype_dict['formats'].append((data_type, data_shape))
            self.dtype_dict['defaults'].append(value)

    def __dtype(self, num_elements):
        '''
        Returns the dtype of the datablock
        add_block_info is only used for blocks
        '''
        # create a new dictionnary to create an 'array of structure of array'
        dtype_aosoa_dict = {}
        dtype_aosoa_dict['names'] = []
        dtype_aosoa_dict['formats'] = []

        for field_name in self.dtype_dict['names']:
            dtype_aosoa_dict['names'].append(field_name)

        for field_format in self.dtype_dict['formats']:
            field_type = field_format[0]
            field_shape = field_format[1]

            # modify the shape to store data as 'array of structure of array'
            # x becomes (num_elements, x)
            # (x, y, ...) becomes (num_elements, x, y, ...)
            new_field_shape = None
            if field_shape == 1:
                # The coma after num_elements is essential
                # In case field_shape == num_elements == 1,
                # it guarantees an array will be produced and not a single value
                new_field_shape = (num_elements,)
            else:
                list_shape = list(field_shape)
                list_shape.insert(0, num_elements)
                new_field_shape = (list_shape)

            dtype_aosoa_dict['formats'].append((field_type, new_field_shape))

        # add block info
        dtype_aosoa_dict['names'].append('blockInfo_numElements')
        dtype_aosoa_dict['formats'].append(np.int64)

        return np.dtype(dtype_aosoa_dict, align=True)

    def initialize(self, num_elements):
        '''
        Initialize blocks and return new block ids
        '''
        self.blocks.clear()
        return self.append(num_elements)

    def append(self, num_elements : int):
        '''
        Initialize blocks and return new element ids
        '''
        block_ids = []
        block_dtype = self.__dtype(self.block_size)

        num_fields = len(self.dtype_dict['names'])
        if num_fields == 0:
            return

        block_id = len(self.blocks)
        global_element_id = self.compute_num_elements()

        n_blocks = math.ceil(num_elements / self.block_size)
        for block_index in range(n_blocks):

            # allocate memory and blockInfo
            block_data = np.zeros(1, dtype=block_dtype)[0] # a scalar

            begin_index = block_index * self.block_size
            block_n_elements = min(self.block_size, num_elements-begin_index)
            block_data['blockInfo_numElements'] = block_n_elements

            # set default values
            for field_id, default_value in enumerate(self.dtype_dict['defaults']):
                block_data[field_id][:] = default_value

            # set ID if available
            if 'ID' in block_data.dtype.names:
                block_data_ID = block_data['ID']
                for block_node_id in range(block_n_elements):
                    na.set_node_id(block_data_ID[block_node_id], global_element_id, block_id, block_node_id)
                    global_element_id += 1

            block_ids.append(block_id)
            block_id += 1
            self.blocks.append(block_data)

        return block_ids

    def remove(self, block_ids = []):
        if not block_ids:
            return

        if 'ID' in self.dtype_dict['names']:
            raise ValueError("ID channel used by this datablock. Another datablock might references this one => cannot delete")

        for block_id in sorted(block_ids, reverse=True):
            del(self.blocks[block_id])

    def isEmpty(self):
        return len(self.blocks)==0

    '''
    Temporary Logic
    Lock/unlock functions to switch between list and tuple
    Numba-0.46.0 doesnt support properly list or numba.typed.list
    Tuple works better so far
    '''
    def lock(self):
        if isinstance(self.blocks, list):
            self.blocks = tuple(self.blocks)

    def unlock(self):
        if isinstance(self.blocks, tuple):
            self.blocks = list(self.blocks)

    '''
    Vectorize Functions on blocks
    '''
    @staticmethod
    def __take_from_id(iterable, block_ids=[]):
        for block_id in block_ids:
            yield iterable[block_id]

    def get_blocks(self, block_ids = []):
        if block_ids:
            return DataBlock.__take_from_id(self.blocks, block_ids)

        return self.blocks

    def compute_num_elements(self, block_ids = []):
        num_elements = 0
        for block_data in self.get_blocks(block_ids):
            num_elements += block_data['blockInfo_numElements']
        return num_elements

    def copyto(self, field_name, values, block_ids = []):
        num_elements = 0

        for block_data in self.get_blocks(block_ids):
            begin_index = num_elements
            block_n_elements = block_data['blockInfo_numElements']
            num_elements += block_n_elements
            end_index = num_elements
            np.copyto(block_data[field_name][0:block_n_elements], values[begin_index:end_index])

    def fill(self, field_name, value, block_ids = []):
        for block in self.get_blocks(block_ids):
            block[field_name].fill(value)

    def flatten(self, field_name, block_ids = []):
        '''
        Convert block of array into a single array
        '''
        field_id = self.dtype_dict['names'].index(field_name)
        field_dtype = self.dtype_dict['formats'][field_id]

        num_elements = self.compute_num_elements(block_ids)
        result = np.empty(num_elements, field_dtype)

        num_elements = 0
        for block_data in self.get_blocks(block_ids):
            begin_index = num_elements
            block_n_elements = block_data['blockInfo_numElements']
            num_elements += block_n_elements
            end_index = num_elements
            np.copyto(result[begin_index:end_index], block_data[field_id][0:block_n_elements])

        return result
