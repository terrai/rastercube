"""
Testing utilities
"""
import os
import unittest
import numpy as np
import rastercube.jgrid as jgrid
from numpy.testing import assert_array_equal


def get_rastercube_dir():
    import rastercube
    return os.path.dirname(rastercube.__file__)


def get_testdata_dir():
    assert 'TERRAI_TEST_DATA' in os.environ, 'You must define TERRAI_TEST_DATA'
    datadir = os.environ['TERRAI_TEST_DATA']
    assert os.path.exists(datadir), 'testdata dir %s does not exist' % datadir
    return datadir


class TerraiTest(unittest.TestCase):
    def setUp(self):
        # Point the test to the test data directory
        os.environ['TERRAI_DATA'] = get_testdata_dir()
