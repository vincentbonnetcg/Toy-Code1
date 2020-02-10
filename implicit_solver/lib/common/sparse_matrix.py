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

    def get_num_entries_per_row(self):
        num_entries_per_row = np.zeros(self.num_rows, dtype=int)
        for row_id in range(self.num_rows):
            num_entries_per_row[row_id] = len(self.dict_indices[row_id])

        return num_entries_per_row

    def sparse_matrix(self):
        num_entries_per_row = self.get_num_entries_per_row()
        total_entries = np.sum(num_entries_per_row)
        min_entry_index = 0 # an entry exists in [0,0] due to mass matrix

        # allocate the sparse matrix
        column_indices = np.zeros(total_entries, dtype=int)
        data = np.zeros((total_entries, self.block_size, self.block_size))
        idx = 0
        for row_id in range(self.num_rows):
            for column_id, matrix in sorted(self.dict_indices[row_id].items()):
                column_indices[idx] = column_id
                data[idx] = matrix
                idx += 1

        row_indptr = np.zeros(self.num_rows+1, dtype=int)
        row_indptr[0] = min_entry_index
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
