"""
Script to extract NDVI dates to a CSV.
This is useful to synchronize dates between multiple jgrids
"""
import os
import argparse
import numpy as np
import rastercube.utils as utils
import rastercube.datasources.modis as modis


parser = argparse.ArgumentParser(description='Create NDVI/QA jgrids from HDF')
parser.add_argument('--tile', type=str, required=True,
                    help='tile name (e.g. h17v07)')
parser.add_argument('--srcdir', type=str, required=False,
                    help='source directory containing '
                         'HDF files, organised in per-year subdirectories'
                         '(e.g. $RASTERCUBE_DATA/0_input/MOD13Q1.005/HDF/LAT/)')
parser.add_argument('--outfile', type=str, required=False,
                    help='Out CSV file containing the dates ')
parser.add_argument('--satellite', type=str, choices=['aqua', 'terra'],
                    default=None)


if __name__ == '__main__':
    args = parser.parse_args()

    tilename = args.tile
    hdf_dir = args.srcdir
    if hdf_dir is None:
        hdf_dir = utils.get_modis_hdf_dir()

    satellite = None
    if args.satellite is not None:
        if args.satellite == 'aqua':
            satellite = 'myd13q1'
        elif args.satellite == 'terra':
            satellite = 'mod13q1'

    outfile = args.outfile

    hdf_files = modis.ndvi_hdf_for_tile(tilename, hdf_dir, satellite=satellite)
    assert len(hdf_files) > 0, 'No matching HDF files found'
    print len(hdf_files), ' HDF files in srcdir'
    hdf_files = sorted(hdf_files, key=lambda t: t[1])

    dates = np.array([utils.format_date(t[1]) for t in hdf_files])

    print dates
    print len(dates), ' dates'
    if outfile is not None:
        np.savetxt(outfile, dates, fmt='%s')
