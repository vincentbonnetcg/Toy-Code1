"""
@author: Vincent Bonnet
@description : Run all tests
"""

import unittest

import code_gen_tests as gen_tests
import datablock_tests as db_tests
import geometry_tests as geo_tests
import numba_tests as numba_tests

if __name__ == '__main__':
    unittest.main(gen_tests.Tests())
    unittest.main(db_tests.Tests())
    unittest.main(geo_tests.Tests())
    unittest.main(numba_tests.Tests())
