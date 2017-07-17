"""
A jGrid is a georeferenced nD array (n >= 3) chunked into fractions along its
x/y axes AND the time axis (and NOT chunked over the other axes).
It is suitable to process pixels as timeseries.

It can be viewed as a nD array (e.g. [width, height, ndates]).

A jGrid consists of a header (a .jghdr3 file, which is just a JSON file) and
of many fractions (stored in a jdata directory alongside the header) which have
a fixed size and are numbered in row-major order according to their position
on the 2D grid.

For the time axis chunks, they are simply numbered. For example, fraction 4390
will have filenames 4390.0, 4390.1, 4390.2, each containing the same xy area
but subsequent slices of the time axis.

Fractions are named '<frac_num>.<frac_time_chunk>.jdata' where frac_num is the
flattened index in the xy grid and frac_time_num is the time chunk.

The (frac_num, frac_time_chunk) tuple is the fraction id.

A jGrid is sparse : if a fraction file does not exist, this means the jGrid
has no data for said fraction.

A jGrid has a dtype (like a numpy array) and each fraction is basically the
binary representation of the nD numpy array

A jGrid also has an associated osr.SpatialReference() which allows to map
and reproject the jGrid. This is stored as WKT in the header.

The jGrid can be stored either on a traditional filesystem or on HDFS where
the fraction size can be set such that a fraction fits in one block, which
leads to good performance for algorithms that can work on a per-fraction basis
"""
from __future__ import division

import os
import sys
import re
import numpy as np
import json
import copy
import cStringIO as StringIO
import rastercube.utils as utils
import pyprind
from osgeo import osr
import rastercube.io as rasterio


def read_frac(fname, hdfs_client=None):
    """
    This returns data or None if the fraction is empty
    """
    if not rasterio.fs_exists(fname, hdfs_client):
        return None
    else:
        if fname.startswith('hdfs://'):
            blob = rasterio.fs_read(fname, hdfs_client)
            return np.load(StringIO.StringIO(blob))
        else:
            # If reading from fs://, we short-circuit fs_read
            return np.load(rasterio.strip_uri_proto(fname, 'fs://'))


def write_frac(fname, data, hdfs_client=None):
    if fname.startswith('hdfs://'):
        buf = StringIO.StringIO()
        np.save(buf, data)
        rasterio.fs_write(fname, buf.getvalue(), hdfs_client)
    else:
        # Short-circuit _fs_read if reading from fs://
        fname = rasterio.strip_uri_proto(fname, 'fs://')
        outdir = os.path.dirname(fname)
        utils.mkdir_p(outdir)
        # If writing to fs://, we short-cirtcuit fs_write
        with open(fname, 'wb') as f:
            np.save(f, data)


# *? and ?? turn on lazy mode (so it first tries to match the frac_num)
FRAC_ID_FROM_FNAME_RE = re.compile(r'^.*?/??(?P<frac_num>\d+)\.(?P<frac_time_chunk>\d+)\.jdata$')


def frac_id_from_fname(fname):
    """
    Given a fraction filename, extracts the fracid
    Returns:
        A tuple (frac_num, frac_time_chunk)
    """
    m = FRAC_ID_FROM_FNAME_RE.match(fname)
    if m is None:
        raise ValueError('Not a fraction filename %s' % fname)
    return (int(m.group('frac_num')), int(m.group('frac_time_chunk')))


def load(gridroot):
    return Header.load(gridroot)


class Header(object):
    """
    Contains jGrid3 metadata and function to load part of the grid.
    """
    def __init__(self, grid_root, width, height, frac_width, frac_height,
                 sr_wkt, dtype, geot, shape, timestamps_ms=None,
                 meta=None, nodataval=None, frac_ndates=None):
        """
        Args:
            grid_root: The grid root (e.g. fs:///worldgrid/ndvi
                or hdfs:///worldgrid/ndvi)
            width, height: the total size of the grid
            frac_width, frac_height : fraction size
            frac_ndates : fraction size along the time axis
            sr_wkt: The spatial reference for the grid, as a WKT understood
                by osr.
            geot: GDAL Affine GeoTransform
                (see http://www.gdal.org/gdal_datamodel.html)
                  Xgeo = geot[0] + Xpixel * geot[1] + Yline * geot[2]
                  Ygeo = geot[3] + Xpixel * geot[4] + Yline * geot[5]
                (so geot[2] and geot[4] shoudl be 0 for north up images
            dtype: A numpy datatype describing the data in the grid
            shape: The full shape of this grid (the first two dimensions are
                redundant with height, width, but if the grid is nD,
                we need the additional dimensions)
            timestamps_ms: A list of timestamp as int in milliseconds. This is
                saved to meta
            meta: A dict of metadata values
            nodataval: The nodata value. This is saved to meta
        """
        assert shape[0] == height
        assert shape[1] == width
        self.shape = shape

        if meta is None:
            meta = {}
        self.meta = meta
        if timestamps_ms is not None:
            self.meta['timestamps_ms'] = timestamps_ms
            assert self.shape[2] == len(timestamps_ms)
        if nodataval is not None:
            self.meta['nodataval'] = nodataval
        self.grid_root = grid_root
        self.width = width
        self.height = height
        self.frac_width = frac_width
        self.frac_height = frac_height
        assert width % frac_width == 0,\
            "width should be a multiple of frac_width"
        assert height % frac_height == 0,\
            "height should be a multiple of frac_height"
        # Note that contrary to frac_width, frac_height, we support frac_ndates
        # NOT dividing exactly the number of timestamps. This is necessary to
        # be able to extend the fractions along the time axis later
        if frac_ndates is not None:
            self.frac_ndates = frac_ndates
        else:
            self.frac_ndates = self.shape[2]

        if self.has_timestamps:
            self.num_dates_fracs = int(
                np.ceil(self.shape[2] / float(self.frac_ndates))
            )
        else:
            self.num_dates_fracs = 1

        self.num_x_fracs = width // frac_width
        self.num_y_fracs = height // frac_height
        self.num_fracs = self.num_x_fracs * self.num_y_fracs
        self.spatialref = osr.SpatialReference()
        self.spatialref.ImportFromWkt(sr_wkt)

        assert np.allclose(geot[2], 0), "geo_t[2] should be 0"
        assert np.allclose(geot[4], 0), "geo_t[4] should be 0"
        self.geot = geot
        self.dtype = np.dtype(dtype)

        wgs84_sr = osr.SpatialReference()
        wgs84_sr.ImportFromEPSG(4326)

        self.wgs84_to_sr = osr.CoordinateTransformation(
            wgs84_sr, self.spatialref)
        self.sr_to_wgs84 = osr.CoordinateTransformation(
            self.spatialref, wgs84_sr)

    @property
    def has_timestamps(self):
        return 'timestamps_ms' in self.meta

    @property
    def timestamps_ms(self):
        if self.has_timestamps:
            return np.array(self.meta['timestamps_ms'])
        else:
            return None

    @property
    def nodataval(self):
        if 'nodataval' in self.meta:
            return self.meta['nodataval']
        else:
            return None

    def copy(self, root=None, dtype=None, shape=None, nodataval=None,
             meta=None, frac_ndates=None):
        """
        Return a copy of this header, with a possibly different root/dtype
        """
        if root is None:
            root = self.grid_root
        if dtype is None:
            dtype = self.dtype
        if shape is None:
            shape = self.shape
        if nodataval is None:
            nodataval = self.nodataval
        if meta is None:
            meta = copy.deepcopy(self.meta)
        if frac_ndates is None:
            frac_ndates = self.frac_ndates
        return Header(
            root, self.width, self.height, self.frac_width, self.frac_height,
            self.spatialref.ExportToWkt(), dtype, self.geot, shape, meta=meta,
            nodataval=nodataval, frac_ndates=frac_ndates)

    def geot_for_xyfrom(self, xy_from):
        """
        Given a (x, y) tuple, returns the geotransform that has its
        top left coordinate at said pixel. This is useful in conjunction
        with the xy_from reported from load_slice
        """
        # Xgeo = geot[0] + Xpixel' * geot[1] + Yline' * geot[2]
        # Ygeo = geot[3] + Xpixel' * geot[4] + Yline' * geot[5]
        #
        # Let Xpixel' = Xpixel + xy_from[0]
        #     Yline'  = Yline + xy_from[1]
        #
        # (and xy_from is constant across pixels)
        # Then, we have to modify geot[0] and geot[3] as follow :
        #   geot'[0] = geot[0] + xy_from[0] * geot[1] + xy_from[1] * geot[2]
        #   geot'[3] = geot[3] + xy_from[0] * geot[4] + xy_from[1] * geot[5]
        #
        new_geot = copy.deepcopy(self.geot)
        new_geot[0] += xy_from[0] * self.geot[1] + xy_from[1] * self.geot[2]
        new_geot[3] += xy_from[0] * self.geot[4] + xy_from[1] * self.geot[5]
        return new_geot

    def xy2latlng(self, xy):
        """
        Returns the latlng for the top left of the pixel at xy
        """
        assert len(xy) == 2
        # (gt[0], gt[3]) is the top left position of the top left pixel
        x, y = xy
        # that is to guarantee that lng = lng2x(x2lng(lng))
        x += 1e-8
        y += 1e-8
        x_geo = self.geot[0] + x * self.geot[1] + y * self.geot[2]
        y_geo = self.geot[3] + x * self.geot[4] + y * self.geot[5]
        lng, lat, _ = self.sr_to_wgs84.TransformPoint(x_geo, y_geo)
        return (lat, lng)

    def latlng2xy(self, latlng):
        lat, lng = latlng
        x_geo, y_geo, _ = self.wgs84_to_sr.TransformPoint(lng, lat)
        # This only works if self.geot[2] == self.geot[4] == 0
        assert np.allclose(self.geot[2], 0)
        assert np.allclose(self.geot[4], 0)
        x = (x_geo - self.geot[0]) / self.geot[1]
        y = (y_geo - self.geot[3]) / self.geot[5]
        return (int(x), int(y))

    def poly_latlng2xy(self, latlngs):
        """
        Convert a list of (lat, lng) tuples
        """
        return map(self.latlng2xy, latlngs)

    def frac_num(self, frac_x, frac_y):
        """Given fraction coordinates, return its frac_num"""
        return frac_y * self.num_x_fracs + frac_x

    def x_start(self, frac_num):
        return (frac_num % self.num_x_fracs) * self.frac_width

    def x_end(self, frac_num):
        return self.x_start(frac_num) + self.frac_width

    def y_start(self, frac_num):
        return (frac_num // self.num_x_fracs) * self.frac_height

    def y_end(self, frac_num):
        return self.y_start(frac_num) + self.frac_height

    def frac_xyranges(self, frac_num):
        return (self.x_start(frac_num), self.x_end(frac_num),
                self.y_start(frac_num), self.y_end(frac_num))

    def frac_time_range(self, frac_time_chunk):
        """
        Returns: The time range this chunk covers as (time_start, time_end),
                 where time_end is exclusive
        """
        t_from = frac_time_chunk * self.frac_ndates
        t_to = min(self.shape[2], t_from + self.frac_ndates)
        return (t_from, t_to)

    def frac_fname(self, frac_id):
        return os.path.join(self.grid_root, 'jdata', '%d.%d.jdata' % frac_id)

    def frac_fnames_for_num(self, frac_num):
        """
        Returns the list of filenames for all the date slices of this fraction
        """
        fnames = []
        for frac_d in xrange(self.num_dates_fracs):
            frac_id = (frac_num, frac_d)
            fnames.append(self.frac_fname(frac_id))
        return fnames

    def load_frac_by_num(self, frac_num, t_from=None, t_to=None,
                         hdfs_client=None):
        """
        Load a fraction given its frac_num and a date range
        """
        if t_from is None:
            t_from = 0
        if t_to is None:
            t_to = self.shape[2]

        ndates = t_to - t_from

        data = np.zeros([self.frac_height, self.frac_width, ndates],
                        dtype=self.dtype)
        # Fill with nodata if we have
        if self.nodataval is not None:
            data[:] = self.nodataval

        d_from = t_from // self.frac_ndates
        d_to = t_to // self.frac_ndates + 1

        for d in range(d_from, d_to):
            frac_t_range = self.frac_time_range(d)
            # Compute the time slice we should take from the fraction
            frac_t_from = max(t_from - frac_t_range[0], 0)
            frac_t_to = min(t_to - frac_t_range[0],
                            frac_t_range[1] - frac_t_range[0])
            assert frac_t_to >= 0
            if frac_t_to - frac_t_from == 0:
                continue

            slice_t_from = frac_t_from + frac_t_range[0] - t_from
            # This correctly handles the truncated time axis case
            slice_t_to = frac_t_to + frac_t_range[0] - t_from
            assert slice_t_from >= 0
            assert slice_t_to >= 0

            # sanity check
            assert slice_t_to - slice_t_from == frac_t_to - frac_t_from

            frac_id = (frac_num, d)

            frac_data = self.load_frac(
                frac_id,
                return_none=True,
                slice=((0, self.frac_height),
                       (0, self.frac_width),
                       (frac_t_from, frac_t_to)),
                hdfs_client=hdfs_client
            )
            if frac_data is not None:
                data[:, :, slice_t_from:slice_t_to] = frac_data

        return data

    def load_frac(self, frac_id, slice=None, hdfs_client=None,
                  return_none=False):
        """
        Load a single fraction.
        This returns data or None if the fraction is empty
        Args:
            slice: A tuple of tuple ((ymin, ymax), (xmin, xmax), (tmin, tmax))
                   specifying the slice to load (in FRACTION coords). If None,
                   defaults to the whole fraction
            return_none: Wether to return None if the fraction is empty
                         Otherwise, returns an empty array
        Returns:
            An array of the shape of slice
        """
        assert len(frac_id) == 2, "frac_id should be (frac_num, frac_t_chunk)"
        if slice is None:
            slice = (
                (0, self.frac_height),
                (0, self.frac_width),
                (0, self.frac_ndates)
            )
        data = read_frac(self.frac_fname(frac_id), hdfs_client)
        if data is not None:
            return data[slice[0][0]:slice[0][1],
                        slice[1][0]:slice[1][1],
                        slice[2][0]:slice[2][1]]
        else:
            if return_none:
                return None
            else:
                height = slice[1] - slice[0]
                width = slice[3] - slice[2]
                ndates = slice[5] - slice[4]
                data = np.zeros(
                    [height, width, ndates] + list(self.shape[3:]),
                    dtype=self.dtype
                )
                return data

    def write_frac_by_num(self, frac_num, data, hdfs_client=None):
        """
        Write all the dates for a single fraction
        """
        assert data.shape[2] == self.shape[2],\
            "You must provide a fraction with all dates to write_frac_by_num"
        assert np.dtype(data.dtype) == self.dtype

        # Write each date slice frac
        for frac_d in xrange(self.num_dates_fracs):
            t1, t2 = self.frac_time_range(frac_d)
            frac_id = (frac_num, frac_d)
            d_data = data[:, :, t1:t2]
            self.write_frac(frac_id, d_data)

    def write_frac(self, frac_id, data, hdfs_client=None):
        """
        Write a single fraction
        """
        assert len(frac_id) == 2, "frac_id should be (frac_num, frac_t_chunk)"
        assert np.dtype(data.dtype) == self.dtype
        # Protect against fraction bigger than frac_ndates
        assert data.shape[2] <= self.frac_ndates, \
            'Corrupted fraction %s, shape[2] is %d, header frac_ndates=%d' % (
                str(frac_id), data.shape[2], self.frac_ndates)
        write_frac(self.frac_fname(frac_id), data, hdfs_client)

    def write_all(self, data):
        """
        Given an array representing the whole grid, write it to disk
        """
        assert np.dtype(data.dtype) == self.dtype
        assert data.shape[:3] == (self.height, self.width, self.shape[2])

        self.save()

        for frac_x in xrange(self.num_x_fracs):
            for frac_y in xrange(self.num_y_fracs):
                for frac_d in xrange(self.num_dates_fracs):
                    frac_num = self.frac_num(frac_x, frac_y)
                    x1, x2, y1, y2 = self.frac_xyranges(frac_num)
                    t1, t2 = self.frac_time_range(frac_d)
                    frac_id = (frac_num, frac_d)
                    self.write_frac(frac_id, data[y1:y2, x1:x2, t1:t2])

    def frac_for_xy(self, x, y):
        """
        Returns the fraction number that will contains the point (x, y)
        """
        assert 0 <= x < self.width
        assert 0 <= y < self.height
        frac_y = int(np.floor(y / self.frac_height))
        frac_x = int(np.floor(x / self.frac_width))
        frac_num = frac_y * self.num_x_fracs + frac_x
        return frac_num

    def fracs_for_rect_xy(self, xy_from, xy_to):
        """
        Returns the list of fraction covering the given area
        (start is included, not end - like numpy)
        """
        # We subtract 1 so that if, for example, frac_width is 50 and
        # x_to is 150, we do not get the third fraction (this x_to is excluded)
        # In all the other cases, this doesn't change anything
        frac_min_x = int(np.floor(xy_from[0] / self.frac_width))
        frac_max_x = int(np.floor((xy_to[0] - 1) / self.frac_width))
        frac_min_y = int(np.floor(xy_from[1] / self.frac_height))
        frac_max_y = int(np.floor((xy_to[1] - 1) / self.frac_height))

        fracs = []
        # Need to add +1 here because we want to be inclusive on fractions
        for frac_x in xrange(frac_min_x, frac_max_x + 1):
            for frac_y in xrange(frac_min_y, frac_max_y + 1):
                frac_num = frac_y * self.num_x_fracs + frac_x
                fracs.append(frac_num)
        return list(set(fracs))

    def load_slice_xy(self, xy_from, xy_to, t_from=None, t_to=None,
                      progressbar=False):
        """
        Load a subset of the grid corresponding to the given rectangle
        (start is included, not end - like numpy)

        Returns:
            data : the data for the requested subrect
        """
        if t_from is None:
            t_from = 0
        if t_to is None:
            t_to = self.shape[2]

        assert self.in_bounds_xy(xy_from)
        assert self.in_bounds_xy(xy_to)
        assert 0 <= t_from < self.shape[2]
        assert 0 <= t_to <= self.shape[2] and t_to > t_from

        fracs = self.fracs_for_rect_xy(xy_from, xy_to)
        sys.stdout.flush()

        slice_width = xy_to[0] - xy_from[0]
        slice_height = xy_to[1] - xy_from[1]
        slice_time = t_to - t_from

        d_from = t_from // self.frac_ndates
        d_to = t_to // self.frac_ndates + 1

        data = np.zeros([slice_height, slice_width, slice_time],
                        dtype=self.dtype)
        # Fill with nodata if we have
        if self.nodataval is not None:
            data[:] = self.nodataval

        nfracs = len(fracs)
        if progressbar:
            bar = pyprind.ProgBar(nfracs)
        for i, frac in enumerate(fracs):
            # Frac start/end in grid coords
            frac_start = (self.x_start(frac), self.y_start(frac))
            frac_end = (self.x_end(frac), self.y_end(frac))

            # Compute the slice of fraction we should take (in grid coords)
            grid_fx1 = max(frac_start[0], xy_from[0])
            grid_fx2 = min(frac_end[0], xy_to[0])
            grid_fy1 = max(frac_start[1], xy_from[1])
            grid_fy2 = min(frac_end[1], xy_to[1])

            # Now, assign the slice of fraction to our grid slice
            slice_fx1 = grid_fx1 - xy_from[0]
            slice_fx2 = grid_fx2 - xy_from[0]
            slice_fy1 = grid_fy1 - xy_from[1]
            slice_fy2 = grid_fy2 - xy_from[1]

            frac_fx1 = grid_fx1 - frac_start[0]
            frac_fx2 = grid_fx2 - frac_start[0]
            frac_fy1 = grid_fy1 - frac_start[1]
            frac_fy2 = grid_fy2 - frac_start[1]

            for d in range(d_from, d_to):
                frac_t_range = self.frac_time_range(d)
                # Compute the time slice we should take from the fraction
                frac_t_from = max(t_from - frac_t_range[0],
                                  0)
                frac_t_to = min(t_to - frac_t_range[0],
                                frac_t_range[1] - frac_t_range[0])
                assert frac_t_to >= 0
                if frac_t_to - frac_t_from == 0:
                    continue

                slice_t_from = frac_t_from + frac_t_range[0] - t_from
                # This correctly handles the truncated time axis case
                slice_t_to = frac_t_to + frac_t_range[0] - t_from
                assert slice_t_from >= 0
                assert slice_t_to >= 0

                # sanity check
                assert slice_t_to - slice_t_from == frac_t_to - frac_t_from

                frac_id = (frac, d)

                frac_data = self.load_frac(
                    frac_id,
                    return_none=True,
                    slice=((frac_fy1, frac_fy2),
                           (frac_fx1, frac_fx2),
                           (frac_t_from, frac_t_to))
                )
                if frac_data is not None:
                    data[slice_fy1:slice_fy2, slice_fx1:slice_fx2,
                         slice_t_from:slice_t_to] = frac_data

            if progressbar:
                bar.update()
        return data

    def load_slice_latlng(self, tl_latlng, br_latlng, t_from=None, t_to=None):
        """
        Load a subset of the grid corresponding to the given rectangle
        (start and end are inclusive) and a given timeslice

        Returns:
            data : the data for the requested subrect
            xy_from : the position of data[0,0] in the grid
        """
        assert tl_latlng[0] > br_latlng[0]
        assert tl_latlng[1] < br_latlng[1]

        xy_from = self.latlng2xy(tl_latlng)
        xy_to = self.latlng2xy(br_latlng)
        assert self.in_bounds_xy(xy_from)
        assert self.in_bounds_xy(xy_to)
        data = self.load_slice_xy(xy_from, xy_to, t_from, t_to)
        return data, xy_from

    def in_bounds_xy(self, xy):
        return 0 <= xy[0] < self.width and 0 <= xy[1] < self.height

    def list_available_fractions(self, hdfs_client=None):
        """
        Returns the list of available (existing) fractions ids.
        Returns:
            a list of tuple (frac_num, time_chunk)
        """
        data_dir = os.path.join(self.grid_root, 'jdata')
        if not rasterio.fs_exists(data_dir, hdfs_client):
            return []
        else:
            fractions = rasterio.fs_list(data_dir, hdfs_client)
            # fractions is a list of fractions filenames (e.g. 14123.jdata)
            fractions = [frac_id_from_fname(fname) for fname in fractions
                         if fname.endswith('jdata')]
            return fractions

    def list_available_fracnums(self, **kwargs):
        """
        Returns a list of available frac nums
        """
        fracs = self.list_available_fractions(**kwargs)
        # extract the frac_num
        return sorted(set(list([f[0] for f in fracs])))

    def to_dict(self):
        d = {
            'width': self.width,
            'height': self.height,
            'fracWidth': self.frac_width,
            'fracHeight': self.frac_height,
            'fracNDates': self.frac_ndates,
            'spatialRefWKT': self.spatialref.ExportToWkt(),
            'dtype': self.dtype.str,
            'geot': self.geot,
            'shape': self.shape,
            'meta': self.meta
        }
        return d

    @staticmethod
    def from_dict(grid_root, d):
        return Header(
            grid_root=grid_root,
            width=d['width'],
            height=d['height'],
            frac_width=d['fracWidth'],
            frac_height=d['fracHeight'],
            frac_ndates=d['fracNDates'],
            dtype=d['dtype'],
            sr_wkt=d['spatialRefWKT'],
            geot=d['geot'],
            meta=d['meta'],
            shape=d['shape'],
        )

    def save(self, hdfs_client=None):
        fname = os.path.join(self.grid_root, 'header.jghdr3')
        blob = json.dumps(self.to_dict())
        rasterio.fs_write(fname, blob, hdfs_client)

    @staticmethod
    def exists(grid_root, hdfs_client=None):
        fname = os.path.join(grid_root, 'header.jghdr3')
        return rasterio.fs_exists(fname, hdfs_client)

    @staticmethod
    def load(grid_root, hdfs_client=None):
        fname = os.path.join(grid_root, 'header.jghdr3')
        d = json.loads(rasterio.fs_read(fname, hdfs_client))
        grid_root = os.path.dirname(fname)
        return Header.from_dict(grid_root, d)
