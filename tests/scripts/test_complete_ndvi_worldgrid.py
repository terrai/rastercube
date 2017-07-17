import os
import sys
import numpy as np
from osgeo import osr, gdal
import tempfile
import pytest
import unittest
import subprocess
import shutil
import tests.utils as test_utils
import rastercube.jgrid as jgrid
import rastercube.jgrid.utils as jgrid_utils
import rastercube.utils as utils
import rastercube.datasources.modis as modis
import rastercube.gdal_utils as gdal_utils
from numpy.testing import assert_array_equal


def assert_grids_same(root1, root2):
    """
    Asserts that two jgrids are the same (same header, same data). This assert
    that the two grids have the same x/y chunking, but NOT necessarily the
    same date chunking
    """
    h1 = jgrid.load(root1)
    h2 = jgrid.load(root2)

    assert np.all(h1.timestamps_ms == h2.timestamps_ms)
    assert np.all(h1.shape == h2.shape)
    assert h1.num_fracs == h2.num_fracs

    fracs1 = h1.list_available_fracnums()
    fracs2 = h2.list_available_fracnums()
    assert np.all(fracs1 == fracs2)

    for frac_num in fracs1:
        data1 = h1.load_frac_by_num(frac_num)
        data2 = h2.load_frac_by_num(frac_num)
        assert_array_equal(data1, data2)


@pytest.mark.usefixtures("setup_env")
def test_complete_ndvi_worldgrid(tempdir):
    """
    Note that this is a quite long test

    Tests the complete_ndvi_worldgrid script by comparing the following :
    - Create a worldgrid using all the ndvi_dates (4 dates) in the testdata
      (grid1)
    - Create a worldgrid with ndvi_dates.2.csv, then append the 2 other dates
      using complete_ndvi_worldgrid
        - Once with frac_ndates = 3 (grid2)
        - Once with frac_ndates = 2 (grid3)
    - Create a worldgrid with ndvi_dates.3.csv, then append the 1 other date
      using complete_ndvi_worldgrid
        - With frac_ndates = 2 (grid4)
        - With frac_ndates = 4 (grid5)
    All 5 grids should contain the exact same data
    """
    create_script = os.path.join(test_utils.get_rastercube_dir(),
                                 'scripts', 'create_ndvi_worldgrid.py')
    complete_script = os.path.join(test_utils.get_rastercube_dir(),
                                   'scripts', 'complete_ndvi_worldgrid.py')

    dates_csv = os.path.join(utils.get_data_dir(), '1_manual',
                             'ndvi_dates.csv')
    dates_csv_2 = os.path.join(utils.get_data_dir(), '1_manual',
                               'ndvi_dates.2.csv')
    dates_csv_3 = os.path.join(utils.get_data_dir(), '1_manual',
                               'ndvi_dates.3.csv')

    def create(rootdir, frac_ndates, dates_csv):
        print 'Creating in %s' % rootdir
        cmd = [
            sys.executable,
            create_script,
            '--tile=h29v07',
            '--noconfirm',
            '--worldgrid=%s' % rootdir,
            '--frac_ndates=%d' % frac_ndates,
            '--dates_csv=%s' % dates_csv,
            # speed things up
            '--test_limit_fractions=2'
        ]
        subprocess.check_call(cmd)

    def complete(rootdir, dates_csv):
        print 'Completing %s' % rootdir
        cmd = [
            sys.executable,
            complete_script,
            '--tile=h29v07',
            '--noconfirm',
            '--worldgrid=%s' % rootdir,
            '--dates_csv=%s' % dates_csv
        ]
        subprocess.check_call(cmd)

    roots = [os.path.join(tempdir, 'grid%d' % n) for n in range(5)]

    create(roots[0], frac_ndates=2, dates_csv=dates_csv)

    create(roots[1], frac_ndates=3, dates_csv=dates_csv_2)
    complete(roots[1], dates_csv)
    # Complete must be idempotent
    complete(roots[1], dates_csv)

    create(roots[2], frac_ndates=2, dates_csv=dates_csv_2)
    complete(roots[2], dates_csv)

    create(roots[3], frac_ndates=2, dates_csv=dates_csv_3)
    complete(roots[3], dates_csv)

    create(roots[4], frac_ndates=3, dates_csv=dates_csv_3)
    complete(roots[4], dates_csv)

    for i in range(1, 5):
        for dsname in ['ndvi', 'qa']:
            assert_grids_same(os.path.join(roots[0], dsname),
                              os.path.join(roots[i], dsname))
