import numpy as np
import random
import unittest
import tempfile
import shutil
import rastercube.jgrid.jgrid3 as jgrid3
from numpy.testing import assert_array_equal


WGS84_WKT = """
GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.01745329251994328,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]
"""

MODIS_SIN_WKT = """
PROJCS["unnamed",
    GEOGCS["Unknown datum based upon the custom spheroid",
        DATUM["Not specified (based on custom spheroid)",
            SPHEROID["Custom spheroid",6371007.181,0]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433]],
    PROJECTION["Sinusoidal"],
    PARAMETER["longitude_of_center",0],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["Meter",1]]
"""


class HeaderTests(unittest.TestCase):
    def setUp(self):
        self.grid_root = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.grid_root)

    def test_latlng2xy(self):
        # Earthexplorer is useful
        # http://earthexplorer.usgs.gov/
        # geotransform for h19v08
        geot = (1111950.519667, 231.65635826374995, 0.0,
                1111950.519667, 0.0, -231.65635826395834)
        width, height, timestamps = 4800, 4800, [1, 2]
        header = jgrid3.Header(
            self.grid_root, width=width, height=height,
            shape=[height, width, len(timestamps)],
            frac_width=200, frac_height=200, sr_wkt=MODIS_SIN_WKT,
            dtype=np.float32, geot=geot, timestamps_ms=timestamps)
        #print header.xy2latlng((0, 0))
        #print header.xy2latlng((header.width, 0))
        #print header.xy2latlng((0, header.height))
        #print header.xy2latlng((header.width, header.height))
        min_lat, min_lng = header.xy2latlng((0, header.height))
        max_lat, max_lng = header.xy2latlng((header.width, 0))
        for i in xrange(10):
            lat = random.uniform(min_lat, max_lat)
            lng = random.uniform(min_lng, max_lng)
            xy = header.latlng2xy((lat, lng))
            lat2, lng2 = header.xy2latlng(xy)
            # Since x, y are rounded to int, we can have an error of the
            # angle size of one pixel
            tol = (max_lng - min_lng) / 4800.
            assert np.allclose(lat, lat2, atol=tol)
            assert np.allclose(lng, lng2, atol=tol)

    def _create_test_grid_header1(self):
        # A 190x130 grid with frac_width=19, frac_height=5
        #
        # This means 10 horiz fractions and 26 vert fractions
        # with truncated fractions at the borders
        mod_pix_size = (231.65, -231.65)
        geot = (-1441428.76, mod_pix_size[0], 0,
                -480362.62, 0, mod_pix_size[1])
        width = 200
        height = 180
        timestamps = [1, 2]
        meta = {'meta1' : 'foobar'}
        header = jgrid3.Header(
            self.grid_root, width=width, height=height,
            shape=[height, width, len(timestamps)],
            frac_width=50, frac_height=60, dtype=np.float32, sr_wkt=WGS84_WKT,
            geot=geot, timestamps_ms=timestamps, meta=meta)
        assert header.num_x_fracs == 4
        assert header.num_y_fracs == 3
        assert header.num_fracs == 12
        return header

    def test_load_save(self):
        header = self._create_test_grid_header1()
        header.save()
        header2 = jgrid3.Header.load(self.grid_root)

        assert_array_equal(header.timestamps_ms, header2.timestamps_ms)
        self.assertEquals(header.width, header2.width)
        self.assertEquals(header.height, header2.height)
        self.assertEquals(header.frac_width, header2.frac_width)
        self.assertEquals(header.frac_height, header2.frac_height)
        self.assertEquals(header.dtype, header2.dtype)
        self.assertTrue(header.spatialref.IsSame(header2.spatialref))
        self.assertEquals(header.meta['meta1'], header2.meta['meta1'])

    def test_write_load_frac(self):
        """Test read/write of single fraction"""
        frac_width = 6
        frac_height = 8
        ndates = 2
        data = np.random.uniform(size=(frac_height, frac_width, ndates))

        geot = (0, 1, 0,
                0, 0, 1)

        width = 60
        height = 48
        timestamps = [1, 2]
        header = jgrid3.Header(
            self.grid_root, width=width, height=height,
            shape=(height, width, len(timestamps)),
            frac_width=frac_width, frac_height=frac_height,
            dtype=data.dtype, sr_wkt=WGS84_WKT, geot=geot,
            timestamps_ms=timestamps)
        frac_id = (0, 1)
        header.write_frac(frac_id, data)
        data2 = header.load_frac(frac_id)

        assert data.dtype == data2.dtype
        assert np.allclose(data, data2, atol=1e-3)

    def test_frac_for_xy(self):
        header = self._create_test_grid_header1()
        frac_num = header.frac_for_xy(0, 0)
        assert frac_num == 0
        frac_num = header.frac_for_xy(49, 59)
        assert frac_num == 0
        frac_num = header.frac_for_xy(49, 60)
        assert frac_num == 4
        frac_num = header.frac_for_xy(189, 129)
        assert frac_num == 11

    def test_frac_for_rect(self):
        header = self._create_test_grid_header1()
        fracs = header.fracs_for_rect_xy((50, 60), (149, 119))
        assert set(fracs) == set([5, 6])

        # The end is exclusive, so this is the same as previous
        fracs = header.fracs_for_rect_xy((50, 60), (150, 120))
        assert set(fracs) == set([5, 6])

        fracs = header.fracs_for_rect_xy((50, 60), (151, 120))
        assert set(fracs) == set([5, 6, 7])

        fracs = header.fracs_for_rect_xy((50, 60), (190, 120))
        assert set(fracs) == set([5, 6, 7])

        fracs = header.fracs_for_rect_xy((50, 60), (150, 130))
        assert set(fracs) == set([5, 6, 9, 10])

        fracs = header.fracs_for_rect_xy((49, 60), (149, 119))
        assert set(fracs) == set([4, 5, 6])

        fracs = header.fracs_for_rect_xy((150, 120), (189, 129))
        assert set(fracs) == set([11])

    def test_load_slice(self):
        """Test reading a slice of a grid"""
        frac_width = 19
        frac_height = 5
        width = 190
        height = 130
        ndates = 11
        data = np.random.uniform(size=(height, width, ndates))

        geot = (0, 1, 0,
                0, 0, 1)

        timestamps = list(np.arange(ndates))
        header = jgrid3.Header(
            self.grid_root, width=width, height=height,
            shape=[height, width, len(timestamps)],
            frac_width=frac_width, frac_height=frac_height,
            frac_ndates=3, dtype=data.dtype, sr_wkt=WGS84_WKT, geot=geot,
            timestamps_ms=timestamps)
        header.write_all(data)

        def _do_xy(xy_from, xy_to):
            data2 = header.load_slice_xy(xy_from, xy_to)
            data_slice = data[xy_from[1]:xy_to[1], xy_from[0]:xy_to[0]]
            assert np.allclose(data_slice, data2, atol=1e-3)

        _do_xy((50, 60), (149, 119))
        _do_xy((49, 60), (149, 119))
        _do_xy((49, 60), (150, 120))
        _do_xy((150, 120), (189, 129))

    def test_load_slice_time(self):
        """Test reading a slice of a grid but only a specific time slice"""
        frac_width = 19
        frac_height = 5
        width = 190
        height = 130
        ndates = 11
        data = np.random.uniform(size=(height, width, ndates))

        geot = (0, 1, 0,
                0, 0, 1)

        timestamps = list(np.arange(ndates))
        header = jgrid3.Header(
            self.grid_root, width=width, height=height,
            shape=[height, width, len(timestamps)],
            frac_width=frac_width, frac_height=frac_height,
            frac_ndates=5, dtype=data.dtype, sr_wkt=WGS84_WKT, geot=geot,
            timestamps_ms=timestamps)
        header.write_all(data)

        def _do_xy(xy_from, xy_to, t_from, t_to):
            data2 = header.load_slice_xy(xy_from, xy_to, t_from=t_from,
                                         t_to=t_to)
            data_slice = data[xy_from[1]:xy_to[1], xy_from[0]:xy_to[0],
                              t_from:t_to]
            assert np.allclose(data_slice, data2, atol=1e-3)

        _do_xy((50, 60), (149, 119), 0, 5)
        _do_xy((49, 60), (149, 119), 5, 11)
        _do_xy((49, 60), (150, 120), 0, 11)
        _do_xy((150, 120), (189, 129), 3, 4)


    def test_load_slice_latlng(self):
        frac_width = 19
        frac_height = 5
        width = 190
        height = 130
        ndates = 2
        data = np.random.uniform(size=(height, width, ndates))

        geot = (0, 1, 0,
                0, 0, -1)

        timestamps = [1, 2]
        header = jgrid3.Header(
            self.grid_root, width=width, height=height,
            shape=[height, width, len(timestamps)],
            frac_width=frac_width, frac_height=frac_height,
            dtype=data.dtype, sr_wkt=WGS84_WKT, geot=geot,
            timestamps_ms=timestamps)
        header.write_all(data)

        # In this case, max_lat > min_lat, because the lat axis growth in
        # the opposite (down to up) than the y axis (up to down)
        min_lat, min_lng = header.xy2latlng((0, 0))
        max_lat, max_lng = header.xy2latlng((header.width, header.height))
        pix_size_deg = ((max_lat - min_lat) / float(height),
                        (max_lng - min_lng) / float(width))

        data2, xy_from = header.load_slice_latlng(
            (min_lat + pix_size_deg[0] * 115,
             min_lng + pix_size_deg[1] * 1),
            (min_lat + pix_size_deg[0] * 125,
             min_lng + pix_size_deg[1] * 12)
        )
        assert xy_from == (1, 115)
        assert data2.shape[:2] == (10, 11)
        data_slice = data[115:125, 1:12]
        assert np.allclose(data_slice, data2, atol=1e-3)


class UtilsTests(unittest.TestCase):
    def test_frac_id_from_fname(self):
        examples = [
            ((33, 0), '33.0.jdata'),
            ((56, 42), 'hdfs:///test/42/bouh/56.42.jdata'),
            ((4543, 3), '/test/bouh//weroi/weroi/woerwer/4543.3.jdata')
        ]
        for expected_num, fname in examples:
            self.assertEqual(expected_num, jgrid3.frac_id_from_fname(fname))
