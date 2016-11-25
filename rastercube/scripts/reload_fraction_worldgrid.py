"""
Reload a fraction from HDF to HDFS

Example invocation::

    python rastercube/scripts/reload_fraction_worldgrid.py
        --fraction=69283
        --fraction_part=0
        --worldgrid=hdfs:///user/terrai/worldgrid
"""
from __future__ import division

import os
import sys
import time
import argparse
import numpy as np
import rastercube.utils as utils
import rastercube.datasources.modis as modis
import rastercube.jgrid as jgrid
import rastercube.worldgrid.grids as grids
import rastercube.io as io
import rastercube.config as config


parser = argparse.ArgumentParser(description='Reload NDVI/QA jgrids from HDF')

parser.add_argument('--fraction', type=int, required=True,
                    help='fraction number (e.g. 69283)')
parser.add_argument('--fraction_part', type=int, required=True,
                    help='fraction part number (e.g. 0)')
parser.add_argument('--worldgrid', type=str, required=True,
                    help='worldgrid root')


def read_ndvi_qa(hdf_file, i_range, j_range):
    modhdf = modis.ModisHDF(hdf_file)
    ds = modhdf.load_gdal_dataset(modis.MODIS_NDVI_DATASET_NAME)
    x, y = j_range[0], i_range[0]
    w = j_range[1] - j_range[0]
    h = i_range[1] - i_range[0]
    ndvi = ds.ReadAsArray(x, y, w, h)
    ds = None

    ds = modhdf.load_gdal_dataset(modis.MODIS_QA_DATASET_NAME)
    qa = ds.ReadAsArray(x, y, w, h)
    ds = None
    return ndvi, qa


if __name__ == '__main__':
    args = parser.parse_args()
    frac_num = args.fraction
    frac_d = args.fraction_part
    frac_id = (frac_num, frac_d)
    modis_dir = utils.get_modis_hdf_dir()
    worldgrid = args.worldgrid
    ndvi_root = os.path.join(worldgrid, 'ndvi')
    qa_root = os.path.join(worldgrid, 'qa')

    assert jgrid.Header.exists(ndvi_root)

    ndvi_header = jgrid.load(ndvi_root)
    qa_header = jgrid.load(qa_root)

    fname = ndvi_header.frac_fname(frac_id)
    if not io.fs_exists(fname):
        print 'The selected fraction does not exist in HDFS'
        exit(0)

    assert np.all(ndvi_header.timestamps_ms == qa_header.timestamps_ms)

    # Select dates for the requested fraction_part
    start_date_i = ndvi_header.frac_ndates * frac_d
    end_date_i = np.amin([len(ndvi_header.timestamps_ms) - start_date_i, ndvi_header.frac_ndates])
    selected_dates = ndvi_header.timestamps_ms[start_date_i:end_date_i]

    modgrid = grids.MODISGrid()

    # Build a dict of frac_num:tilename
    tiles = config.MODIS_TERRA_TILES
    frac_tilename = {}
    for t_n in tiles:
        h, v = modis.parse_tilename(t_n)
        for f_n in modgrid.get_cells_for_tile(h, v):
            frac_tilename[f_n] = t_n
            
    # Select the tile
    tilename = frac_tilename[frac_num]
    tile_h, tile_v = modis.parse_tilename(tilename)
    i_range, j_range = modgrid.get_cell_indices_in_tile(frac_num, tile_h, tile_v)

    print
    print 'Will reload the following :'
    print 'Fraction:', frac_num
    print 'Part:', frac_d
    print 'tilename:', tilename
    print len(selected_dates), 'dates'
    print

    if not utils.confirm(prompt='Proceed?', resp=True):
        sys.exit(-1)
        
    _start = time.time()

    ## -- Find the filenames of the HDF file for this date and our tile
    hdf_files = modis.ndvi_hdf_for_tile(tilename, modis_dir)
    hdf_files = {ts:fname for (fname, ts) in hdf_files}

    ndvi = np.empty((ndvi_header.frac_height, ndvi_header.frac_width, len(selected_dates)), dtype='int16')
    qa = np.empty((qa_header.frac_height, qa_header.frac_width, len(selected_dates)), dtype='uint16')
    
    for (t_i, ts) in enumerate(selected_dates):
        fname = hdf_files[ts]
        print t_i, fname
        
        new_ndvi, new_qa = read_ndvi_qa(fname, i_range, j_range)
        ndvi[:,:,t_i] = new_ndvi
        qa[:,:,t_i] = new_qa
        
    ndvi_header.write_frac(frac_id, ndvi)
    qa_header.write_frac(frac_id, qa)

    print 'Processed %d, took %.02f [s]' % (frac_num, time.time() - _start)

    
    

