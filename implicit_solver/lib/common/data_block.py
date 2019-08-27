"""
@author: Vincent Bonnet
@description : Array of Structures of Arrays (AoSoA)
Example :
data = DataBlock()
data.add_field("field_a", np.float, 1)
data.add_field("field_b", np.float, (2, 2))
data.initialize(10)
print(data.field_a)
print(data.field_b)
"""

import numpy as np
import keyword

class DataBlock:

    def __init__(self):
        self.reset()

    def reset(self):
        '''
        Reset the datablock (reset data and datatype)
        '''
        # Data
        self.num_elements = 0
        self.data = None
        self.blocks = [] # TODO - replace data
        # Datatype
        self.dtype_dict = {}
        self.dtype_dict['names'] = [] # list of names
        self.dtype_dict['formats'] = [] # list of tuples (data_type, data_shape)
        self.dtype_dict['defaults'] = [] # list of default values (should match formats)

    def is_allocated(self):
        '''
        Return whether the datablock is allocated
        '''
        if self.data:
            return True
        return False

    def clear(self):
        '''
        Clear the data on the datablock (it doesn't reset the datatype)
        '''
        self.num_elements = 0
        self.data = None
        self.blocks.clear()

    def add_field_from_class(self, class_type):
        self.add_field_from_instance(class_type())

    def add_field_from_instance(self, inst):
        for name, value in inst.__dict__.items():
            self.__add_field_from_value(name, value)

    def add_field(self, name, data_type=np.float, data_shape=1):
        self.__add_field_from_type(name, data_type, data_shape)

    def __check_before_add(self, name):
        '''
        Raise exception if 'name' cannot be added
        '''
        if self.is_allocated():
            raise ValueError("Cannot add fields after initialized DataBlock")

        if keyword.iskeyword(name):
            raise ValueError("field name cannot be a keyword: " + name)

        if name in self.dtype_dict['names']:
            raise ValueError("field name already used : " + name)

    def __add_field_from_type(self, name, data_type=np.float, data_shape=1):
        '''
        Add a new field to the data block
        '''
        self.__check_before_add(name)

        zero_value = None

        if data_shape == 1:
            zero_value = data_type(0.0)
        else:
            zero_value = np.zeros(data_shape, data_type)

        self.dtype_dict['names'].append(name)
        self.dtype_dict['formats'].append((data_type, data_shape))
        self.dtype_dict['defaults'].append(zero_value)

    def __add_field_from_value(self, name, value):
        '''
        Add a new field to the data block
        '''
        self.__check_before_add(name)

        data_type = None
        data_shape = None

        if np.isscalar(value):
            data_type = type(value)
            data_shape = 1
        else:
            data_type = value.dtype.type
            data_shape = value.shape

        self.dtype_dict['names'].append(name)
        self.dtype_dict['formats'].append((data_type, data_shape))
        self.dtype_dict['defaults'].append(value)

    def fill(self, field_name, value):
        if not self.is_allocated():
            return

        # set data
        self.data[field_name].fill(value)
        # set blocks
        for block in self.blocks:
            block[field_name].fill(value)

    def initialize_from_array(self, array):
        '''
        Allocate the fields from the array
        SLOW but generic
        '''
        num_constraints = len(array)
        self.initialize(num_constraints)
        for index, element in enumerate(array):
            for attribute_name, attribute_value in element.__dict__.items():
                if attribute_name in self.dtype_dict['names']:
                    field_index = self.dtype_dict['names'].index(attribute_name)
                    self.data[field_index][index] =  getattr(element, attribute_name)

    def initialize(self, num_elements):
        '''
        Allocate the fields
        '''
        self.clear()
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
                if field_shape == 1:
                    # The coma after self.num_elements is essential
                    # In case field_shape == num_elements == 1,
                    # it guarantees an array will be produced and not a single value
                    new_field_shape = (self.num_elements,)
                else:
                    new_field_shape = (self.num_elements, field_shape)
            else:
                list_shape = list(field_shape)
                list_shape.insert(0, self.num_elements)
                new_field_shape = tuple(list_shape)

            dtype_aosoa_dict['formats'].append((field_type, new_field_shape))

        # allocate memory
        aosoa_dtype = np.dtype(dtype_aosoa_dict, align=True)
        self.data = np.zeros(1, dtype=aosoa_dtype)[0] # a scalar

        # Set default values
        for field_index, default_value in enumerate(self.dtype_dict['defaults']):
            self.data[field_index][:] = default_value

    def __len__(self):
        return self.num_elements

    def __getattr__(self, item):
        '''
        Access a specific field from data
        '''
        if item == "data" or self.is_allocated() is None:
           raise AttributeError

        if item in self.dtype_dict['names']:
            field_index = self.dtype_dict['names'].index(item)
            return self.data[field_index]

        raise AttributeError
