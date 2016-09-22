"""
Script to print statistics about downloaded HDF files. This prints NDVI tiles
with incomplete dates.

Example invocation::

    python rastercube/scripts/ndvi_hdf_stats.py

"""
import argparse
import numpy as np
import rastercube.utils as utils
import rastercube.datasources.modis as modis
import collections


parser = argparse.ArgumentParser(description='Create NDVI/QA jgrids from HDF')
parser.add_argument('--hdfdir', type=str,
                    help='source directory containing '
                         'HDF files, organised in per-year subdirectories'
                         '(e.g. $RASTERCUBE_DATA/0_input/MOD13Q1.005/HDF/LAT/)')


if __name__ == '__main__':
    args = parser.parse_args()

    hdf_dir = args.hdfdir
    if hdf_dir is None:
        hdf_dir = utils.get_modis_hdf_dir()

    hdf_files = modis.ndvi_list_hdf(hdf_dir)
    ntiles = len(hdf_files)
    print ntiles, ' tiles in srcdir'
    print np.sum([len(l) for l in hdf_files.values()]), ' HDF files'

    for tile_name in sorted(hdf_files.keys()):
        print '-- %s' % tile_name
        tile_files = hdf_files[tile_name]
        print len(tile_files), ' files'

    # Now, find the dates which are NOT in all tiles
    # Maps timestamp to tiles containing this timestamp
    date2tiles = collections.defaultdict(lambda: [])
    for tile_name in hdf_files:
        for fname, timestampms in hdf_files[tile_name]:
            date2tiles[timestampms].append(tile_name)

    print '\n\n---- Dates that are not present for all tiles'
    all_tiles = set(hdf_files.keys())
    missing_times = date2tiles.keys()
    for timestampms in sorted(missing_times):
        tiles = date2tiles[timestampms]
        if len(tiles) != ntiles:
            missing_tiles = all_tiles.difference(tiles)
            date = utils.date_from_timestamp_ms(timestampms)
            date_ymd = date.strftime('%Y_%m_%d')
            doy = date.strftime('%j')
            print date_ymd, '(doy=', doy, ') missing : ',\
                ' '.join(missing_tiles)
