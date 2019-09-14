"""
@author: Vincent Bonnet
@description : Run all tests
"""

import code_gen_tests as gen_tests
import datablock_tests as db_tests
import unittest

if __name__ == '__main__':
    unittest.main(gen_tests.TestCodeGeneration())
    unittest.main(db_tests.TestDataBlock())
