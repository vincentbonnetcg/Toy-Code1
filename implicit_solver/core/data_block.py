"""
@author: Vincent Bonnet
@description : Hierarchical Tiling Layout to store the data
This structure is also called an 'array of structure of array'
Example :
data = DataBlock()
data.add_field("field_a", np.float, 1)
data.add_field("field_b", np.float, (2, 2))
data.initialize(10)
print(data.field_a)
print(data.field_b)
"""

import warnings
import numpy as np

class DataBlock:

    def __init__(self):
        self.num_elements = 0
        # Data
        self.data = None
        # Dictionnary to describe the dtype
        self.dtype_dict = {}
        self.dtype_dict['names'] = []
        self.dtype_dict['formats'] = []

    def add_field(self, name, data_type=np.float, data_shape=1):
        '''
        Add a new field to the data block
        '''
        if self.data is None:
            self.dtype_dict['names'].append(name)
            self.dtype_dict['formats'].append((data_type, data_shape))
        else:
            warnings.warn('Cannot addField on an initialized DataBlock')

    def dtype(self):
        return np.dtype(self.dtype_dict)

    def initialize(self, num_elements):
        '''
        Allocate the fields
        '''
        if self.data is None:
            self.num_elements = num_elements
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
                new_field_shape = tuple()
                if np.isscalar(field_shape):
                    new_field_shape = (self.num_elements, field_shape)
                else:
                    list_shape = list(field_shape)
                    list_shape.insert(0, self.num_elements)
                    new_field_shape = tuple(list_shape)

                dtype_aosoa_dict['formats'].append((field_type, new_field_shape))

            # allocate memory
            aosoa_dtype = np.dtype(dtype_aosoa_dict)
            self.data = np.zeros(1, dtype=aosoa_dtype)[0] # a scalar
        else:
            warnings.warn('DataBlock is already allocated')

    def set_attribute_to_object(self, obj):
        if self.data is None:
            return None

        for field_index, field_name in enumerate(self.dtype_dict['names']):
            setattr(obj, field_name, self.data[field_index])

    def __getattr__(self, item):
        '''
        Access a specific field from data
        '''
        if item == "data" or self.data is None:
           raise AttributeError

        if item in self.dtype_dict['names']:
            field_index = self.dtype_dict['names'].index(item)
            return self.data[field_index]

        raise AttributeError
