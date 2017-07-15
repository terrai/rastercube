"""
Integrates MODIS images for a new date to an existing worldgrid

Example invocation::

    python rastercube/scripts/complete_ndvi_worldgrid.py
        --tile=h10v09
        --worldgrid=hdfs:///user/terrai/worldgrid
        --dates_csv=$RASTERCUBE_TEST_DATA/1_manual/ndvi_dates.3.csv
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
import joblib


parser = argparse.ArgumentParser(description='Create NDVI/QA jgrids from HDF')

parser.add_argument('--tile', type=str, required=True,
                    help='tile name (e.g. h17v07, all)')
parser.add_argument('--noconfirm', action='store_true',
                    help='Skip confirmation')
parser.add_argument('--worldgrid', type=str, required=True,
                    help='worldgrid root')
parser.add_argument('--nworkers', type=int, default=None,
                    help='Number of workers (by default all)')
parser.add_argument('--modis_dir', type=str, required=False,
                    help='directory where input MODIS files are stored')
parser.add_argument('--dates_csv', type=str, default=None,
                    help='The dates that must be included in the grid'
                         'see scripts/ndvi_collect_dates.py')


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


def complete_frac(frac_num, ndvi_root, qa_root, frac_tilename, tilename_fileindex):
    """
    Given a frac_num, will make sure it contains data for all dates in
    ndvi_header.timestamps_ms

    Args:
        hdf_files: A dict mapping timestamp to HDF filename
    """
    _start = time.time()
    modgrid = grids.MODISGrid()
    ndvi_header = jgrid.load(ndvi_root)
    qa_header = jgrid.load(qa_root)

    tilename = frac_tilename[frac_num]
    tile_h, tile_v = modis.parse_tilename(tilename)
    hdf_files = tilename_fileindex[tilename]

    d_from = 0
    d_to = ndvi_header.shape[2] // ndvi_header.frac_ndates + 1

    # Find the most recent existing fraction and the most recent timestamp
    for frac_d in range(d_from, d_to)[::-1]:
        frac_id = (frac_num, frac_d)
        fname = ndvi_header.frac_fname(frac_id)
        if io.fs_exists(fname):
            break

    # Read the data of the most recent fraction in HDFS
    ndvi = jgrid.read_frac(ndvi_header.frac_fname(frac_id))
    qa = jgrid.read_frac(qa_header.frac_fname(frac_id))

    assert np.all(ndvi.shape == qa.shape)

    # Compute the index of the last date in HDFS
    most_recent_t = frac_d * ndvi_header.frac_ndates + ndvi.shape[2]

    i_range, j_range = modgrid.get_cell_indices_in_tile(frac_num, tile_h, tile_v)

    # At this point, we just have to complete with the missing dates
    for t in range(most_recent_t, len(ndvi_header.timestamps_ms)):
        ts = ndvi_header.timestamps_ms[t]
        fname = hdf_files[ts]

        new_ndvi, new_qa = read_ndvi_qa(fname, i_range, j_range)

        if ndvi.shape[2] == ndvi_header.frac_ndates:
            # Write a complete fraction
            frac_id = (frac_num, frac_d)
            ndvi_header.write_frac(frac_id, ndvi)
            qa_header.write_frac(frac_id, qa)
            # Prepare variables for a new fraction
            frac_d += 1
            ndvi = new_ndvi[: , :, None]
            qa = new_qa[:, :, None]
        else:
            # TODO: If we end up completing multiple dates, we could preallocate
            # But for now, this is unlikely (we'll complete with the most
            # recent data)
            ndvi = np.concatenate([ndvi, new_ndvi[:, :, None]], axis=2)
            qa = np.concatenate([qa, new_qa[:, :, None]], axis=2)

        assert np.all(ndvi.shape == qa.shape)

    # Write last incomplete fraction
    frac_id = (frac_num, frac_d)
    ndvi_header.write_frac(frac_id, ndvi)
    qa_header.write_frac(frac_id, qa)

    print 'Processed %d, appended %d dates, took %.02f [s]' % (frac_num,
           len(ndvi_header.timestamps_ms) - most_recent_t,
           time.time() - _start)

    sys.stdout.flush()


def set_header_timestamps(header, new_dates_ms):
    header.meta['timestamps_ms'] = new_dates_ms
    header.shape[2] = len(new_dates_ms)
    header.num_dates_fracs = int(
        np.ceil(header.shape[2] / float(header.frac_ndates))
    )


if __name__ == '__main__':
    args = parser.parse_args()
    tilename = args.tile
    modis_dir = utils.get_modis_hdf_dir()
    worldgrid = args.worldgrid
    ndvi_root = os.path.join(worldgrid, 'ndvi')
    qa_root = os.path.join(worldgrid, 'qa')
    nworkers = args.nworkers
    modgrid = grids.MODISGrid()
    tiles = config.MODIS_TERRA_TILES

    assert jgrid.Header.exists(ndvi_root)

    print 'Loading headers...'
    ndvi_header = jgrid.load(ndvi_root)
    qa_header = jgrid.load(qa_root)

    assert np.all(ndvi_header.timestamps_ms == qa_header.timestamps_ms)

    assert args.dates_csv is not None

    print 'Reading dates from csv...'
    # -- Verify that dates_csv match the header
    dates = np.genfromtxt(args.dates_csv, dtype=str)
    dates_ms = sorted([utils.parse_date(d) for d in dates])
    header_ndates = len(ndvi_header.timestamps_ms)
    assert np.all(ndvi_header.timestamps_ms == dates_ms[:header_ndates])

    print 'Updating headers...'
    # -- Update the headers with the new timestamps
    set_header_timestamps(ndvi_header, dates_ms)
    ndvi_header.save()
    set_header_timestamps(qa_header, dates_ms)
    qa_header.save()

    print 'Reloading headers...'
    # reload the headers
    ndvi_header = jgrid.load(ndvi_root)
    qa_header = jgrid.load(qa_root)
    assert np.all(ndvi_header.timestamps_ms == dates_ms)
    assert np.all(qa_header.timestamps_ms == dates_ms)
    assert np.all(ndvi_header.list_available_fracnums() ==
                  qa_header.list_available_fracnums())


    if tilename == 'all':
        print 'All fractions selected'
        fractions = ndvi_header.list_available_fracnums()
    else:
        tiles = [tilename]
        print 'Selecting fractions of tile', tilename, '...'
        # -- Figure out the fractions we have to update
        t_h, t_v = modis.parse_tilename(tilename)
        fractions_tile = modgrid.get_cells_for_tile(t_h, t_v)
        assert np.all(ndvi_header.list_available_fracnums() ==
                      qa_header.list_available_fracnums())
        fractions = np.intersect1d(fractions_tile,
                                   ndvi_header.list_available_fracnums())

    if len(fractions) == 0:
        print 'No fractions to process... Terminating'
        sys.exit(0)

    print
    print 'Will append the following :'
    print 'NDVI grid root : %s' % ndvi_root
    print 'QA grid root : %s' % qa_root
    print 'tilename : %s' % tilename
    print 'Number of fractions : %d' % len(fractions)
    print 'nworkers : %s' % str(nworkers)
    print

    if not args.noconfirm:
        if not utils.confirm(prompt='Proceed?', resp=True):
            sys.exit(0)

    print 'Building dictionary of fraction:tilename...'
    # Build a dictionary frac_num:tilename
    frac_tilename = {}
    for t_n in tiles:
        h, v = modis.parse_tilename(t_n)
        for f_n in modgrid.get_cells_for_tile(h, v):
            frac_tilename[f_n] = t_n

    print 'Building dict of dict of tilename->file index'
    # Find MODIS files for every tile
    # This builds a dict of dict that maps each tilename to a dict of
    # (timestamp, filename) like
    # {'h29v07' : {950832000000: 'date1.hdf', 952214400000: 'date2.hdf'}}
    tilename_fileindex = {}
    for t_n in tiles:
        hdf_f = modis.ndvi_hdf_for_tile(t_n, modis_dir)
        hdf_f = {ts:fname for (fname, ts) in hdf_f}
        tilename_fileindex[t_n] = hdf_f

    # Launch the process
    if nworkers is not None and nworkers > 1:
        print 'Using joblib.Parallel with nworkers=%d' % nworkers
        joblib.Parallel(n_jobs=nworkers)(
            joblib.delayed(complete_frac)(frac_num, ndvi_root, qa_root,
                                          frac_tilename, tilename_fileindex)
            for frac_num in fractions
        )
    else:
        for frac_num in fractions:
            complete_frac(frac_num, ndvi_root, qa_root,
                          frac_tilename, tilename_fileindex)

