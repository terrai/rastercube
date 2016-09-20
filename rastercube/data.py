import numpy as np
from json import loads
import os
import bisect
import datetime


def data_path():
    path = os.environ['TERRAI_DATA']
    assert path != '', 'You must define $TERRAI_DATA'
    return path


def data_filepath(fname):
    return os.path.join(data_path(), fname)


def load_ndvi_timestamps():
    import rastercube.utils as utils
    fname = os.path.join(data_path(), '1_manual', 'ndvi_dates.csv')
    dates = np.genfromtxt(fname, dtype=str)
    dates_ms = sorted([utils.parse_date(d) for d in dates])
    return dates_ms


def load_terrai_zip(fname, abs_path=False):
    """
    Loads a terra-i exported zip.
    If abs_path is False, fname is relative to $TERRAI_DATA,
    otherwise it is absolute
    """
    if abs_path:
        d = np.load(fname)
    else:
        d = np.load(data_filepath(fname))

    # jgrid has x as the first dimension, we want y
    data = np.rollaxis(d['data'], 1, 0)
    mask = np.rollaxis(d['mask'], 1, 0)
    meta = loads(d['meta.json'])
    return data, mask, meta


def show_info(metadata):
    """
    Shows the boundaries of data loaded with the function load_terrai_zip using
    2 decimals.
    """
    right_lng = ('%.2f' % (metadata["bottom_right_lng"],))
    left_lng = ('%.2f' % (metadata["top_left_lng"],))
    top_lat = ('%.2f' % (metadata["top_left_lat"],))
    bottom_lat = ('%.2f' % (metadata["bottom_right_lat"],))
    while len(left_lng) < 8:
        left_lng = " " + left_lng
    print "              ", top_lat
    print "         .---------------."
    print "         |               |"
    print "         |               |"
    print left_lng, "|               |", right_lng
    print "         |               |"
    print "         |               |"
    print "         '---------------'"
    print "              ", bottom_lat


def reshape_trmm(data_trmm, meta_trmm, meta_ndvi):
    """
    Cuts and reshapes a trmm array to match a ndvi array
    """

    trmm_indexes = np.array([bisect.bisect_left(meta_trmm["timestamps"], t) for
                            t in meta_ndvi["timestamps"]])
    trmm_indexes = trmm_indexes[trmm_indexes < data_trmm.shape[2]]

    lat_index_from = np.floor((meta_trmm["top_left_lat"] -
                              meta_ndvi["top_left_lat"]) /
                              meta_trmm["cellsize"])
    lat_index_to = np.ceil((meta_trmm["top_left_lat"] -
                           meta_ndvi["bottom_right_lat"]) /
                           meta_trmm["cellsize"])
    long_index_from = np.floor((meta_ndvi["top_left_lng"] -
                               meta_trmm["top_left_lng"]) /
                               meta_trmm["cellsize"])
    long_index_to = np.ceil((meta_ndvi["bottom_right_lng"] -
                            meta_trmm["top_left_lng"]) /
                            meta_trmm["cellsize"])

    data_trmm_16 = np.zeros((lat_index_to - lat_index_from, long_index_to -
                            long_index_from, len(trmm_indexes), 16))
    for j in range(16):
        data_trmm_16[:, :, :, j] = data_trmm[lat_index_from:lat_index_to,
                                             long_index_from:long_index_to,
                                             [i + j for i in trmm_indexes]]
    print "TRMM reshaped:", data_trmm_16.shape

    return data_trmm_16


def join_ndvi_trmm(data_ndvi, data_trmm, row, column, date, scale=120,
                   number_of_samples=25):
    """
    Creates an observation joining number_of_samples previous samples of ndvi
    and the 16 previous samples of trmm to the current ndvi.
    """
    return np.concatenate((data_ndvi[row, column, (date - number_of_samples):
                                     date], data_trmm[np.floor(row / scale),
                                                      np.floor(column / scale),
                                                      date, :],
                           [data_ndvi[row, column, date]]))


def compute_pixel_from_lat_long(lat, lng, meta):
    dlng = lng - meta['top_left_lng']
    dlat = meta['top_left_lat'] - lat
    xpix = int(dlng / meta['cellsize'])
    ypix = int(dlat / meta['cellsize'])
    return xpix, ypix


def compute_lat_long_from_pixel(x, y, meta):
    rlng = x * meta["cellsize"]
    rlat = y * meta["cellsize"]
    lng = meta["top_left_lng"] + rlng
    lat = meta["top_left_lat"] - rlat
    return lat, lng


def ts2date(timestamp):
    # terra-i timestamps are milliseconds
    dt = datetime.datetime.fromtimestamp(timestamp / 1000.0)
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def get_indices_for_year_from_header(year, header):
    indices = list()
    for i, timestamp in enumerate(header.timestamps_ms):
        st = ts2date(timestamp)
        if int(st[:4]) == year:
            indices.append(i)
    assert len(indices) > 0, "No detections for this year"
    return np.array(indices)


def get_values_for_year_from_header(year, array, header):
    assert array.dtype != np.bool, "Use get_detections_for_year_from_header +\
    instead!"
    indices = get_indices_for_year_from_header(year, header)

    temp = array[:, :, indices]

    return np.logical_or.reduce(temp, axis=2)


def get_detections_for_year_from_header(year, array, header):
    assert array.dtype == np.bool, "Use get_values_for_year_from_header +\
    instead!"
    indices = get_indices_for_year_from_header(year, header)

    temp = array[:, :, indices]
    temp = np.logical_or.reduce(temp, axis=2)

    # Gets the previous detections (if any)
    previous = array[:, :, 0:indices[0]]

    if np.count_nonzero(previous) > 0:
        previous = np.logical_or.reduce(previous, axis=2)

        # And removes them from the detections of this year
        temp[previous] = False

    return temp
