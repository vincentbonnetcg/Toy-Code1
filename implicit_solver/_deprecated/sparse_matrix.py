"""
@author: Vincent Bonnet
@description : sparse matrix helpers
"""

from abc import ABCMeta, abstractmethod

import numpy as np
import scipy as sc

class BaseSparseMatrix(object, metaclass=ABCMeta):
    '''
    Interface for sparse matrix
    '''
    def __init__(self, num_rows, num_columns, block_size):
        self.num_rows = num_rows
        self.num_columns = num_columns
        self.block_size = block_size

    @abstractmethod
    def add(self, i, j, data):
        pass

    @abstractmethod
    def sparse_matrix(self):
        pass

class BSRSparseMatrix(BaseSparseMatrix):
    '''
    Helper class to build a BSR matrix
    '''
    def __init__(self, num_rows, num_columns, block_size):
        BaseSparseMatrix.__init__(self, num_rows, num_columns, block_size)
        self.dict_indices = [None] * self.num_rows
        for i in range(self.num_rows):
            self.dict_indices[i] = {}

    def add(self, i, j, data):
        value = self.dict_indices[i].get(j, None)
        if value is None:
            self.dict_indices[i][j] = data
        else:
            value += data

    def sparse_matrix(self, num_entries_per_row, column_indices, data):
        row_indptr = np.zeros(self.num_rows+1, dtype=int)
        row_indptr[0] = 0 # minimum entry exists at [0,0] due to mass matrix
        np.add.accumulate(num_entries_per_row, out=row_indptr[1:self.num_rows+1])

        return sc.sparse.bsr_matrix((data, column_indices, row_indptr))

class DebugSparseMatrix(BaseSparseMatrix):
    '''
    Helper class to debug the sparse matrix
    It is just a dense matrix
    '''
    def __init__(self, num_rows, num_columns, block_size):
        BaseSparseMatrix.__init__(self, num_rows, num_columns, block_size)
        self.matrix = np.zeros((self.num_rows * self.block_size,
                                self.num_columns * self.block_size))

    def add(self, i, j, data):
        self.matrix[i*self.block_size:i*self.block_size+self.block_size,
                    j*self.block_size:j*self.block_size+self.block_size] += data

    def sparse_matrix(self):
        return sc.sparse.bsr_matrix(self.matrix, blocksize=(self.block_size, self.block_size))
