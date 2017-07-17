"""
Import the given GLCF tile into the provided worldgrid.

Example invocation::

    python rastercube/scripts/create_glcf_worldgrid.py
        --year 2004
        --tile ML1920
        --worldgrid=hdfs:///user/test/
"""
import os
import sys
import time
import gzip
import shutil
import argparse
import tempfile
import numpy as np
import rastercube.utils as utils
import rastercube.jgrid as jgrid
import rastercube.worldgrid.grids as grids
from osgeo import gdal


parser = argparse.ArgumentParser(description='Create GLCF jgrids from TIF')
parser.add_argument('--tile', type=str, required=True,
                    help='tile name (e.g. ML1920)')
parser.add_argument('--year', type=str, required=False, default=2004)
parser.add_argument('--noconfirm', action='store_true',
                    help='Skip confirmation')
parser.add_argument('--glcf_dir', type=str, required=False,
                    help='directory where input GLCF files are stored')
parser.add_argument('--worldgrid', type=str, required=True,
                    help='worldgrid root')
parser.add_argument('--force_all', action='store_true',
                    help='If True, will recreate all fractions')


def import_glcf_tile(glcf_header, cell_num, tilefile):
    glcfgrid = grids.GLCFGrid()
    glcf_data = np.zeros((glcfgrid.cell_height, glcfgrid.cell_width, 1),
                         dtype=np.uint8)

    with tempfile.NamedTemporaryFile() as f:
        # uncompress
        with gzip.open(tilefile, 'rb') as f_in, open(f.name, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        gdal_ds = gdal.Open(f.name)
        assert gdal_ds is not None, "Failed to open GDAL dataset"
        band = gdal_ds.GetRasterBand(1)
        nodata_val = band.GetNoDataValue()
        print "NoData : ", nodata_val
        glcf_data[:, :, 0] = band.ReadAsArray()

    glcf_header.write_frac((cell_num, 0), glcf_data)

    print 'Finished %d' % cell_num

    return True


if __name__ == '__main__':
    args = parser.parse_args()

    tilename = args.tile
    year = int(args.year)
    glcf_dir = args.glcf_dir
    if glcf_dir is None:
        glcf_dir = utils.get_glcf_tif_dir()

    glcf_grid_root = os.path.join(args.worldgrid, 'glcf/%d' % year)
    worldgrid = grids.GLCFGrid()

    if not jgrid.Header.exists(glcf_grid_root):
        glcf_header = jgrid.Header(
            glcf_grid_root, worldgrid.width,
            worldgrid.height, worldgrid.cell_width, worldgrid.cell_height,
            worldgrid.proj_wkt, dtype=np.uint8, geot=worldgrid.geot,
            shape=(worldgrid.height, worldgrid.width, 1))

        glcf_header.save()

        print 'Saved header in ', glcf_grid_root
    else:
        glcf_header = jgrid.Header.load(glcf_grid_root)

    tif_fname = 'MCD12Q1_V51_LC1.%d.%s' % (year, tilename)
    # the tif is stored in a directory with the same fname as the .tif file
    tif_file = os.path.join(
        glcf_dir, '%d.01.01' % year,
        tif_fname,
        tif_fname + ".tif.gz")
    assert os.path.exists(tif_file), "Couldn't find input TIF %s" % tif_file

    fraction = worldgrid.get_cell_for_tile(tilename)
    if not args.force_all:
        if fraction in glcf_header.list_available_fractions():
            print 'Fraction already exists, use --force_all to force'
            sys.exit(0)

    print
    print 'Will import the following :'
    print 'tilename : %s' % tilename
    print 'fraction : %d' % fraction
    print 'input GLCF dir  : %s' % glcf_dir
    print 'output GLCF grid root : %s' % glcf_grid_root
    print 'year : %d' % year
    print

    if not args.noconfirm:
        if not utils.confirm(prompt='Proceed?', resp=True):
            sys.exit(-1)

    start = time.time()

    import_glcf_tile(glcf_header, fraction, tif_file)

    elapsed = time.time() - start
    print 'Took %f [s]' % elapsed
