"""
Shows how many fractions are available in the worldgrid for a given tile

Example invocation::

    python rastercube/scripts/show_available_fractions.py
        --tile=h10v09
        --worldgrid=hdfs:///user/terrai/worldgrid/
"""
from __future__ import division

import os
import sys
import argparse
import numpy as np
import rastercube.utils as utils
import rastercube.datasources.modis as modis
import rastercube.jgrid as jgrid
import rastercube.worldgrid.grids as grids

parser = argparse.ArgumentParser(description='Create NDVI/QA jgrids from HDF')

parser.add_argument('--tile', type=str, required=True,
                    help='tile name (e.g. h17v07, all)')
parser.add_argument('--worldgrid', type=str, required=True,
                    help='worldgrid root')


if __name__ == '__main__':
    args = parser.parse_args()
    arg_tilename = args.tile
    modis_dir = utils.get_modis_hdf_dir()
    worldgrid = args.worldgrid
    ndvi_root = os.path.join(worldgrid, 'ndvi')
    qa_root = os.path.join(worldgrid, 'qa')

    assert jgrid.Header.exists(ndvi_root)

    print 'Reading HDF headers...'
    ndvi_header = jgrid.load(ndvi_root)
    qa_header = jgrid.load(qa_root)

    assert np.all(ndvi_header.timestamps_ms == qa_header.timestamps_ms)

    if arg_tilename == 'all':
        import rastercube.config as config
        tiles = config.MODIS_TERRA_TILES
    else:
        tiles = [arg_tilename]
        
    print 'Starting...'
    for tilename in tiles:
        print tilename, ':',
        sys.stdout.flush()

        # -- Find the filename of the HDF file for this date and our tile
        print 'Finding files...',
        sys.stdout.flush()
        hdf_files = modis.ndvi_hdf_for_tile(tilename, modis_dir)
        hdf_files = {ts: fname for (fname, ts) in hdf_files}

        # -- Figure out the fractions we have to update
        modgrid = grids.MODISGrid()
        tile_h, tile_v = modis.parse_tilename(tilename)

        print 'Finding fractions...',
        sys.stdout.flush()
        fractions = modgrid.get_cells_for_tile(tile_h, tile_v)
        assert np.all(ndvi_header.list_available_fracnums() ==
                      qa_header.list_available_fracnums())
        fractions = np.intersect1d(fractions,
                                   ndvi_header.list_available_fracnums())

        print '->', len(fractions), 'available'
        sys.stdout.flush()

