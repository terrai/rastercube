"""
This is a script to deal with missing HDF files from MODIS ftp

For some tiles, it might (rarely) happen that there is a missing date.
This script will create a "fake" HDF file for the missing date based on 
a previous HDF of the given tile.

This create a HDF with 0000000000000 as the julian production date.

Example invocation::

    python rastercube/scripts/create_ndvi_worldgrid.py
        --tile=h10v09
        --worldgrid=hdfs:///user/test/
        --dates_csv=$RASTERCUBE_TEST_DATA/1_manual/ndvi_dates.2.csv

"""
import os
import sys
import time
import argparse
import warnings
import pyhdf
import pyhdf.SD
import ctypes
import shutil
import numpy as np
import multiprocessing
import multiprocessing.sharedctypes
import rastercube.utils as utils
import rastercube.datasources.modis as modis
import rastercube.jgrid as jgrid
import rastercube.worldgrid.grids as grids


parser = argparse.ArgumentParser(description="Create a new NDVI worldgrid")

parser.add_argument('--tile', type=str, required=True,
                    help='tile name (e.g. h17v07)')
parser.add_argument('--noconfirm', action='store_true',
                    help='Skip confirmation')
parser.add_argument('--missing_date', type=str, required=True,
                    help='the missing date in YYYYddd format (e.g. 2007137)')
parser.add_argument('--prefix', type=str, required=True,
                    help='satellite type (mod or myd)')
parser.add_argument('--modis_dir', type=str, required=False,
                    help='directory where input MODIS files are stored')


def collect_hdf_files(tilename, hdf_dir):
    # hdf_files contains (full path, timestamp_ms)
    hdf_files = modis.ndvi_hdf_for_tile(tilename, hdf_dir)
    assert len(hdf_files) > 0, 'No matching HDF files found'
    print len(hdf_files), ' HDF files in srcdir'

    return hdf_files


if __name__ == '__main__':
    # Print help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    assert args.prefix.lower() in ['mod', 'myd'], "Prefix must be either mod or myd"

    tilename = args.tile
    modis_dir = args.modis_dir
    if modis_dir is None:
        modis_dir = utils.get_modis_hdf_dir()

    hdf_files = modis.ndvi_hdf_for_tile(tilename, modis_dir)
    # Pick any file as the model
    model_hdf = hdf_files[0][0]

    prefix = args.prefix.upper()
    year = int(args.missing_date[:4])
    doy = int(args.missing_date[4:])

    output_fname = os.path.join(
        modis_dir,
        str(year),
        '%s13Q1.A%d%d.%s.005.0000000000000.hdf' % (prefix, year, doy, tilename)
    )

    assert not os.path.exists(output_fname), "%s already exists" % output_fname

    print 'Using %s as the model HDF' % model_hdf
    print 'year : %d' % year
    print 'doy : %d' % doy
    print 'satellite prefix : %s' % prefix
    print 'output filename : %s' % output_fname

    if not args.noconfirm:
        if not utils.confirm(prompt='Proceed?', resp=False):
            sys.exit(-1)

    shutil.copyfile(model_hdf, output_fname)

    d = pyhdf.SD.SD(output_fname, pyhdf.SD.SDC.WRITE)
    ndvi = d.select('250m 16 days NDVI')
    arr = ndvi.get()
    arr[:] = -3000
    ndvi.set(arr)
    del ndvi

    qa = d.select('250m 16 days VI Quality')
    arr = qa.get()
    arr[:] = 65535
    qa.set(arr)
    del qa
    del d
    print 'Done'
