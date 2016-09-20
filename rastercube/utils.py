"""
Utility functions
"""
import rastercube
import numpy as np
import os
import errno
from datetime import datetime
import calendar
import cPickle as pickle
import pkg_resources
import atexit

# Cleanup tmpdir used by asset_fname on interpreter exit
atexit.register(lambda : pkg_resources.cleanup_resources())

def asset_fname(relpath):
    """
    Gets the filename to an asset relative to the rastercube package root.

    When rastercube is packaged as an egg, you can't access assets using
    os.path.join(rastercube.__file__, 'assets/foo.json') since the egg is a
    zip. So you should use this function.

    See :
    http://peak.telecommunity.com/DevCenter/PythonEggs#accessing-package-resources

    >>> fname = asset_fname('assets/modis_tiles.geojson')
    """
    return pkg_resources.resource_filename(rastercube.__name__, relpath)


def get_data_dir():
    assert 'RASTERCUBE_DATA' in os.environ
    return os.environ['RASTERCUBE_DATA']


def get_worldgrid():
    assert 'RASTERCUBE_WORLDGRID' in os.environ
    return os.environ['RASTERCUBE_WORLDGRID']


def get_modis_hdf_dir():
    """Returns the default directory where we store MODIS HDF files"""
    return os.path.join(get_data_dir(), '0_input', 'MODIS_HDF')


def get_glcf_tif_dir():
    """Returns the default directory where we store MODIS HDF files"""
    return os.path.join(get_data_dir(), '0_input', 'glcf_5.1')


def mkdir_p(path):
    """Create all directory in path. Like mkdir -p"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise


def load_properties(filename):
    properties = {}
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith(';') or len(line) == 0:
                continue
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            properties[key] = value
    return properties


def date_from_timestamp_ms(timestamp_ms):
    date = datetime.utcfromtimestamp(timestamp_ms / 1000.0)
    return date


def format_date(timestamp_ms, sep=None):
    """
    Like jGridUtils.formatDate
    """
    if sep is None:
        sep = '_'
    date = date_from_timestamp_ms(timestamp_ms)
    return date.strftime('%Y{0}%m{0}%d'.format(sep))


def day_to_timestamp_ms(year, month, day):
    """Returns the milliseconds timestamp for the given date"""
    return calendar.timegm(datetime(year, month, day).timetuple()) * 1000


def timestamp_ms_to_doy(timestamp_ms):
    """Convert a timestamp in milliseconds to day-of-year"""
    d = datetime.fromtimestamp(timestamp_ms / 1000.).date()
    return int(d.strftime('%j'))


def parse_date(datestr, sep=None):
    if sep is None:
        sep = '_'
    date = datetime.strptime(datestr, '%Y{0}%m{0}%d'.format(sep))
    timestamp_ms = int(calendar.timegm(date.timetuple()) * 1000.0)
    return timestamp_ms


def confirm(prompt=None, resp=False):
    """
    Prompts for yes or no response from the user. Returns True for yes and
    False for no.

    'resp' should be set to the default value assumed by the caller when
    user simply types ENTER.
    """
    if prompt is None:
        prompt = 'Confirm'

    if resp:
        prompt = '%s [%s]|%s: ' % (prompt, 'y', 'n')
    else:
        prompt = '%s [%s]|%s: ' % (prompt, 'n', 'y')

    while True:
        ans = raw_input(prompt)
        if not ans:
            return resp
        if ans not in ['y', 'Y', 'n', 'N']:
            print 'please enter y or n.'
            continue
        if ans == 'y' or ans == 'Y':
            return True
        if ans == 'n' or ans == 'N':
            return False


def save(fname, obj):
    with open(fname, 'w') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load(fname):
    with open(fname) as f:
        return pickle.load(f)


def index_3d_with_2d(array, indices):
    """
    Given a 3D array a, will index it with a 2D array b that contains,
    the index along the z axis to select.
    This will return a 2D array c where
        c[i,j] = array[i,j,indices[i,j]]

    This ought to be done with choice but is somewhat complicated. Relevant
    stackoverflow discussion: http://stackoverflow.com/a/32090582

    >>> a = np.arange(24).reshape(2, 3, 4)
    >>> a
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
    <BLANKLINE>
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>> b = np.array([[0, 1, 2],
    ...               [3, 0, 1]])
    >>> index_3d_with_2d(a, b)
    array([[ 0,  5, 10],
           [15, 16, 21]])
    """
    assert len(array.shape) == 3
    assert len(indices.shape) == 2
    h, w, d = array.shape
    return array.reshape(-1, d)[np.arange(h * w), indices.reshape(-1)]\
                .reshape(h, w)
