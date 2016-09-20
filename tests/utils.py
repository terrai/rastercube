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
    assert 'RASTERCUBE_TEST_DATA' in os.environ, 'You must define RASTERCUBE_TEST_DATA'
    datadir = os.environ['RASTERCUBE_TEST_DATA']
    assert os.path.exists(datadir), 'testdata dir %s does not exist' % datadir
    return datadir


class RasterCubeTest(unittest.TestCase):
    def setUp(self):
        # Point the datadir to the test datadir
        os.environ['RASTERCUBE_DATA'] = get_testdata_dir()
