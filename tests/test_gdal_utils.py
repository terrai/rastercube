import numpy as np
import random
import unittest
import rastercube.gdal_utils as gdal_utils
from numpy.testing import assert_allclose


class GdalDsFromArrayTest(unittest.TestCase):
    def test_2d(self):
        arr = np.random.rand(15, 17)
        ds = gdal_utils.gdal_ds_from_array(arr)
        assert_allclose(arr.squeeze(), ds.ReadAsArray().squeeze())

    def test_3d(self):
        arr = (np.random.rand(35, 2, 18) * 255).astype(np.int16)
        ds = gdal_utils.gdal_ds_from_array(arr)
        # GDAL ReadAsArray returns (bands, height, width), hence the transpose
        assert_allclose(arr, ds.ReadAsArray().transpose(1, 2, 0))
