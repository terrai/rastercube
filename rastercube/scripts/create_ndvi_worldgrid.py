"""
Import the given MODIS tile into the provided NDVI and QA worldgrid.
Assert that the MODIS tile contains (at least) the requested dates.

Note that this require a .csv file with NDVI dates to import. This file
can be create with the ``ndvi_collect_dates.py`` script.

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
import ctypes
import numpy as np
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
parser.add_argument('--modis_dir', type=str, required=False,
                    help='directory where input MODIS files are stored')
parser.add_argument('--worldgrid', type=str, required=True,
                    help='worldgrid root')
# If we have fractions of 400x400x50 and store int16, we get
# 400 * 400 * 50 * 2 / (1024 * 1024.) = 15MB
parser.add_argument('--frac_ndates', type=int, default=50,
                    help='Size of a chunk along the time axis')
parser.add_argument('--nworkers', type=int, default=5,
                    help='Number of workers (if using multiprocessing)')
parser.add_argument('--dates_csv', type=str, default=None,
                    help='The dates that must be included in the grid'
                         'see scripts/ndvi_collect_dates.py')
parser.add_argument('--test_limit_fractions', type=int, default=None,
                    help='(TESTING ONLY) : Only create the first n fractions')


def collect_hdf_files(tilename, hdf_dir):
    # hdf_files contains (full path, timestamp_ms)
    hdf_files = modis.ndvi_hdf_for_tile(tilename, hdf_dir)
    assert len(hdf_files) > 0, 'No matching HDF files found'
    print len(hdf_files), ' HDF files in srcdir'

    return hdf_files

# ------------------------------------- Shared multiprocessing globals

# Global variable initialize by _mp_init
_mp_ndvi = None
_mp_qa = None


def _mp_init(shared_ndvi, shared_qa):
    global _mp_ndvi, _mp_qa
    _mp_ndvi = shared_ndvi
    _mp_qa = shared_qa


# ------------------------------------- Multiprocess HDF processing

def _real_mp_process_hdf(hdf_file, frac_ti, grid_w, grid_h, frac_ndates):
    """
    Args:
        frac_ti: The time index of the hdf_file in the current frac array
    """
    # ignore the PEP 3118 buffer warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        s_ndvi = np.ctypeslib.as_array(_mp_ndvi)
        s_ndvi.shape = (grid_h, grid_w, frac_ndates)
        s_ndvi.dtype = np.int16
        s_qa = np.ctypeslib.as_array(_mp_qa)
        s_qa.shape = (grid_h, grid_w, frac_ndates)
        s_qa.dtype = np.uint16

    _start = time.time()
    modhdf = modis.ModisHDF(hdf_file)

    # -- ndvi
    _ndvi_start = time.time()
    ds = modhdf.load_gdal_dataset(modis.MODIS_NDVI_DATASET_NAME)
    ds.ReadAsArray(buf_obj=s_ndvi[:, :, frac_ti])
    _ndvi_elapsed = time.time() - _ndvi_start
    del ds

    # -- qa
    _qa_start = time.time()
    ds = modhdf.load_gdal_dataset(modis.MODIS_QA_DATASET_NAME)
    ds.ReadAsArray(buf_obj=s_qa[:, :, frac_ti])
    _qa_elapsed = time.time() - _qa_start
    del ds

    print 'Loading ', os.path.basename(hdf_file),\
        'took %.02f [s] (%.02f ndvi read, %.02f qa)' % (
        time.time() - _start, _ndvi_elapsed, _qa_elapsed)
    sys.stdout.flush()


def _mp_process_hdf(args):
    """
    Wrapper around _mp_process_hdf that correctly handles keyboard
    interrupt
    """
    # TODO: This is supposed to make CTRL-C work but it doesn't
    try:
        _real_mp_process_hdf(*args)
    except (KeyboardInterrupt, SystemExit):
        print "Worker interrupted, exiting..."
        return False


# ------------------------------------- Multiprocess fractions writing

def _real_mp_write_frac(frac_id, grid_w, grid_h, frac_ndates):
    # ignore the PEP 3118 buffer warning
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        s_ndvi = np.ctypeslib.as_array(_mp_ndvi)
        s_ndvi.shape = (grid_h, grid_w, frac_ndates)
        s_ndvi.dtype = np.int16
        s_qa = np.ctypeslib.as_array(_mp_qa)
        s_qa.shape = (grid_h, grid_w, frac_ndates)
        s_qa.dtype = np.uint16

    frac_num, frac_d = frac_id

    i_range, j_range = modgrid.get_cell_indices_in_tile(
        frac_num, tile_h, tile_v)
    frac_ndvi = s_ndvi[i_range[0]:i_range[1], j_range[0]:j_range[1], :]
    frac_qa = s_qa[i_range[0]:i_range[1], j_range[0]:j_range[1], :]

    ndvi_header.write_frac(frac_id, frac_ndvi)
    qa_header.write_frac(frac_id, frac_qa)


def _mp_write_frac(args):
    try:
        _real_mp_write_frac(*args)
    except (KeyboardInterrupt, SystemExit):
        print "Worker interrupted, exiting..."
        return False


if __name__ == '__main__':
    # Print help if no arguments are provided
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    tilename = args.tile
    modis_dir = args.modis_dir
    if modis_dir is None:
        modis_dir = utils.get_modis_hdf_dir()

    test_limit_fractions = args.test_limit_fractions

    nworkers = args.nworkers

    worldgrid = args.worldgrid
    ndvi_grid_root = os.path.join(worldgrid, 'ndvi')
    qa_grid_root = os.path.join(worldgrid, 'qa')

    if not jgrid.Header.exists(ndvi_grid_root):
        assert args.dates_csv is not None

        dates = np.genfromtxt(args.dates_csv, dtype=str)
        dates_ms = sorted([utils.parse_date(d) for d in dates])

        grid = grids.MODISGrid()

        ndvi_header = jgrid.Header(
            grid_root=ndvi_grid_root,
            width=grid.width,
            height=grid.height,
            frac_width=grid.cell_width,
            frac_height=grid.cell_height,
            frac_ndates=args.frac_ndates,
            sr_wkt=grid.proj_wkt,
            dtype=np.int16,
            geot=grid.geot,
            shape=(grid.height, grid.width, len(dates_ms)),
            timestamps_ms=dates_ms,
            nodataval=modis.MODIS_NDVI_NODATA
        )

        ndvi_header.save()

        qa_header = jgrid.Header(
            grid_root=qa_grid_root,
            width=grid.width,
            height=grid.height,
            frac_width=grid.cell_width,
            frac_height=grid.cell_height,
            frac_ndates=args.frac_ndates,
            sr_wkt=grid.proj_wkt,
            dtype=np.uint16,
            geot=grid.geot,
            shape=(grid.height, grid.width, len(dates_ms)),
            timestamps_ms=dates_ms,
            nodataval=modis.MODIS_QA_NODATA
        )

        qa_header.save()

        print 'Saved header in ', ndvi_grid_root, qa_grid_root
    else:
        ndvi_header = jgrid.Header.load(ndvi_grid_root)
        qa_header = jgrid.Header.load(qa_grid_root)
        assert np.all(ndvi_header.timestamps_ms == qa_header.timestamps_ms)

        if args.dates_csv is not None:
            # Verify that dates_csv match the header
            dates = np.genfromtxt(args.dates_csv, dtype=str)
            dates_ms = sorted([utils.parse_date(d) for d in dates])
            assert np.all(ndvi_header.timestamps_ms == dates_ms)
    assert args.frac_ndates == ndvi_header.frac_ndates,\
        "Existing header has different frac_ndates (%d) than requested (%d)" % \
        (ndvi_header.frac_ndates, args.frac_ndates)

    hdf_files = collect_hdf_files(tilename, modis_dir)

    # Verify that we have all necessary timestamps
    header_timestamps = set(ndvi_header.timestamps_ms)
    files_timestamps = set([t[1] for t in hdf_files])
    difference = header_timestamps.difference(files_timestamps)
    assert len(difference) == 0, \
        'difference between available' \
        ' dates and required : %s' % \
        ' '.join([utils.format_date(d) for d in difference])
    # only pick files for which the timestamp has been requested
    hdf_files = filter(lambda f: f[1] in header_timestamps, hdf_files)

    modgrid = grids.MODISGrid()
    tile_h, tile_v = modis.parse_tilename(tilename)
    fractions = modgrid.get_cells_for_tile(tile_h, tile_v)
    if test_limit_fractions is not None:
        # This should only be used for testing as a mean to speed things
        # up by only creating a limited number of fractions
        print 'TEST - Limiting fractions'
        fractions = fractions[:test_limit_fractions]

    grid_w = modgrid.MODIS_tile_width
    grid_h = modgrid.MODIS_tile_height

    max_frac_size_mb = (grid_w * grid_h * ndvi_header.frac_ndates * 2 / (1024. * 1024.))

    print
    print 'Will import the following :'
    print 'tilename : %s' % tilename
    print 'tile_h=%d, tile_v=%d' % (tile_h, tile_v)
    print 'num fractions : %d' % len(fractions)
    print 'input MODIS dir  : %s' % modis_dir
    print 'output NDVI grid root : %s' % ndvi_grid_root
    print 'output QA grid root : %s' % qa_grid_root
    print 'date range : %s' % (utils.format_date(hdf_files[0][1]) + ' - ' +
                               utils.format_date(hdf_files[-1][1]))
    print 'num source hdf files : %d' % len(hdf_files)
    print 'required memory : %d [Mb]' % max_frac_size_mb
    print

    if len(fractions) == 0:
        print 'No fractions to process - terminating'
        sys.exit(0)

    if not args.noconfirm:
        if not utils.confirm(prompt='Proceed?', resp=True):
            sys.exit(-1)

    _start = time.time()

    assert ndvi_header.frac_ndates == qa_header.frac_ndates

    for frac_d in xrange(ndvi_header.num_dates_fracs):

        frac_time_range = np.arange(*ndvi_header.frac_time_range(frac_d))
        frac_ndates = len(frac_time_range)

        # We directly use short for data and let the workers do the conversion
        shared_ndvi = multiprocessing.sharedctypes.RawArray(
            ctypes.c_short, grid_w * grid_h * frac_ndates)
        shared_qa = multiprocessing.sharedctypes.RawArray(
            ctypes.c_short, grid_w * grid_h * frac_ndates)

        pool = multiprocessing.Pool(
            processes=nworkers,
            initializer=_mp_init,
            initargs=(shared_ndvi, shared_qa)
        )
        try:
            # The .get(9999999) are a ugly fix for a python bug where the keyboard
            # interrupt isn't raised depending on when it happens
            # see
            # http://stackoverflow.com/a/1408476

            # 1. Read data
            _read_start = time.time()
            args = []
            for frac_ti, t in enumerate(frac_time_range):
                fname, timestamp = hdf_files[t]
                args.append((fname, frac_ti, grid_w, grid_h, frac_ndates))
            pool.map_async(_mp_process_hdf, args).get(9999999)
            print 'Read took %f [s]' % (time.time() - _read_start)

            # 2. Write fractions
            _write_start = time.time()
            args = []
            for frac_num in fractions:
                frac_id = (frac_num, frac_d)
                args.append((frac_id, grid_w, grid_h, frac_ndates))
            pool.map_async(_mp_write_frac, args).get(9999999)
            print 'Write took %f [s]' % (time.time() - _write_start)

            pool.close()
            pool.join()
        except KeyboardInterrupt:
            print "Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            pool.join()
            sys.exit(-1)

    print 'Took %f [s]' % (time.time() - _start)
