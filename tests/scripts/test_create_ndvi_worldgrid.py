import os
import sys
import numpy as np
from osgeo import osr, gdal
import tempfile
import unittest
import subprocess
import shutil
import rastercube.data as terradata
import tests.utils as test_utils
import rastercube.jgrid as jgrid
import rastercube.jgrid.utils as jgrid_utils
import rastercube.utils as utils
import rastercube.datasources.modis as modis
import rastercube.gdal_utils as gdal_utils
from numpy.testing import assert_array_equal


class CreateNDVIWorldgridTest(test_utils.TerraiTest):
    """
    This is really an integration test because it tests the following :
    - Creates of a NDVI worldgrid from HDF files
    - Writes the NDVI fractions containing the HDF data
    - Loads (using geographical coordinates) the area corresponding to the
      HDF from the created jgrid
    - Verify that the jgrid data matches the HDF data

    This require that jgrid read/write work properly as well as all the jgrid
    georeferencing
    """
    def setUp(self):
        super(CreateNDVIWorldgridTest, self).setUp()
        self.tmpdir = tempfile.mkdtemp()

    def test_create_ndvi_worldgrid(self):
        script = os.path.join(test_utils.get_rastercube_dir(), 'scripts',
                              'create_ndvi_worldgrid.py')
        ndvi_dir = os.path.join(self.tmpdir, 'ndvi')
        qa_dir = os.path.join(self.tmpdir, 'qa')
        dates_csv = os.path.join(terradata.data_path(), '1_manual',
                                 'ndvi_dates.2.csv')
        tile = 'h29v07'
        print 'dates_csv : ', dates_csv
        cmd = [
            sys.executable,
            script,
            '--tile=%s' % tile,
            '--noconfirm',
            '--ndvi_grid_root=%s' % ndvi_dir,
            '--qa_grid_root=%s' % qa_dir,
            '--frac_ndates=1',
            '--dates_csv=%s' % dates_csv
        ]
        try:
            output = subprocess.check_output(cmd)
        except subprocess.CalledProcessError as e:
            print 'Failed running'
            print ' '.join(cmd)

        # Verify that the header has the correct dates
        ndvi_header = jgrid.load(ndvi_dir)
        qa_header = jgrid.load(qa_dir)
        dates = [utils.format_date(ts) for ts in ndvi_header.timestamps_ms]
        self.assertEqual(['2000_02_18', '2000_03_05'], dates)

        self.assertEqual(ndvi_header.num_dates_fracs, 2)

        # Load the HDF and the corresponding date from the jgrid and
        # check for consistency
        hdf_fname = os.path.join(utils.get_modis_hdf_dir(), '2000',
                'MOD13Q1.A2000065.h29v07.005.2008238013448.hdf')
        f = modis.ModisHDF(hdf_fname)
        ndvi_ds = f.load_gdal_dataset(modis.MODIS_NDVI_DATASET_NAME)

        hdf_ndvi = ndvi_ds.ReadAsArray()
        hdf_qa = f.load_gdal_dataset(
                modis.MODIS_QA_DATASET_NAME).ReadAsArray()

        # Load from the jgrid using the lat/lng polygon of the HDF file
        # This means we also test georeferencing
        hdf_poly = gdal_utils.latlng_bounding_box_from_ds(ndvi_ds)
        xy_from, qa, qa_mask, ndvi, ndvi_mask = \
            jgrid_utils.load_poly_latlng_from_multi_jgrids(
                    [qa_header, ndvi_header], hdf_poly)
        self.assertEqual(ndvi.shape[:2], hdf_ndvi.shape)
        # Verify that the jgrid ndvi and the HDF ndvi store the same values
        assert_array_equal(hdf_ndvi, ndvi[:,:,1])
        assert_array_equal(hdf_qa, qa[:,:,1])

    def tearDown(self):
        assert len(self.tmpdir) > 0
        shutil.rmtree(self.tmpdir)


if __name__ == '__main__':
    unittest.main()
