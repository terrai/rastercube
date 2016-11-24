"""
This scripts was made to correct a bug in the script that completes the fractions
with new dates. The original complete_ndvi_worldgrid.py appends new data to the
first file only, so it growed until 715 dates!
This script reads the first file (xxxxx.0.jdata) and truncates the array to have
a size of (frac_width, frac_height, frac_ndates) which should correspond to 
(400, 400, 200)

Example invocation::

    python rastercube/scripts/truncate_first_hdfs_file_to_200.py
        --worldgrid=hdfs:///user/terrai/worldgrid/
"""
from __future__ import division

import os
import sys
import time
import argparse
import numpy as np
import rastercube.jgrid as jgrid
import rastercube.worldgrid.grids as grids
import rastercube.io as io
import joblib

parser = argparse.ArgumentParser(description='Create NDVI/QA jgrids from HDF')

parser.add_argument('--worldgrid', type=str, required=True,
                    help='worldgrid root')
parser.add_argument('--nworkers', type=int, default=None,
                    help='Number of workers (by default all)')

def truncate_frac(frac_num, ndvi_root, qa_root):
    """
    Given a frac_num, will truncate the first hdfs file to have a size of
    (frac_width, frac_height, frac_ndates) which should correspond to
    (400, 400, 200)
    

    """
    _start = time.time()
    ndvi_header = jgrid.load(ndvi_root)
    qa_header = jgrid.load(qa_root)

    frac_d = 0
    frac_id = (frac_num, frac_d)
    ndvi = jgrid.read_frac(ndvi_header.frac_fname(frac_id))
    qa = jgrid.read_frac(qa_header.frac_fname(frac_id))

    # At this point, we just have to truncate the array
    if ndvi is not None:
        if ndvi.shape[2] > ndvi_header.frac_ndates:
            ndvi = ndvi[:,:,0:ndvi_header.frac_ndates]
            ndvi_header.write_frac(frac_id, ndvi)
        else:
            print frac_num, ': NDVI already OK'
    else:
        print frac_num, ': NDVI is None'

    if qa is not None:
        if qa.shape[2] > qa_header.frac_ndates:
            qa = qa[:,:,0:qa_header.frac_ndates]
            qa_header.write_frac(frac_id, qa)
        else:
            print frac_num, ': QA already OK'
    else:
        print frac_num, ': QA is None'

    print 'Processed %d, took %.02f [s]' % (frac_num, time.time() - _start)
    sys.stdout.flush()


if __name__ == '__main__':
    args = parser.parse_args()
    worldgrid = args.worldgrid
    ndvi_root = os.path.join(worldgrid, 'ndvi')
    qa_root = os.path.join(worldgrid, 'qa')
    nworkers = args.nworkers

    assert jgrid.Header.exists(ndvi_root)

    ndvi_header = jgrid.load(ndvi_root)
    qa_header = jgrid.load(qa_root)

    assert np.all(ndvi_header.timestamps_ms == qa_header.timestamps_ms)

    # -- Figure out the fractions we have to update
    assert np.all(ndvi_header.list_available_fracnums() == \
                  qa_header.list_available_fracnums())
    fractions = ndvi_header.list_available_fracnums()

    if len(fractions) == 0:
        print 'No fractions to process - terminating'
        sys.exit(0)

    if nworkers is not None and nworkers > 1:
        print 'Using joblib.Parallel with nworkers=%d' % nworkers
        joblib.Parallel(n_jobs=nworkers)(
            joblib.delayed(truncate_frac)(frac_num, ndvi_root, qa_root)
            for frac_num in fractions
        )
    else:
        for frac_num in fractions:
            truncate_frac(frac_num, ndvi_root, qa_root)

