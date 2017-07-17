"""
Utility functions to deal with MODIS HDF data
"""
import os
import re
import calendar
import gdal
import collections
import numpy as np
from datetime import datetime
import rastercube.datasources.modis_qa as modis_qa


# Lowercase regexp to match MODIS filenames
# See https://lpdaac.usgs.gov/products/modis_products_table/modis_overview
# e.g. mod13q1.a2015065.h17v08.005.2015083062452.hdf
MODIS_NDVI_REGEX = r'^(?P<sattype>m(o|y)d13q1)\.' \
                   r'a(?P<julian_date>\d{7})\.' \
                   r'h(?P<tile_h>\d{2})v(?P<tile_v>\d{2})\.' \
                   r'(?P<coll>\d{3})\.' \
                   r'(?P<julian_prod_date>\d{13})\.' \
                   r'hdf$'

MODIS_NDVI_MATCHER = re.compile(MODIS_NDVI_REGEX)

TILENAME_REGEX = r'^h(?P<tile_h>\d{2})v(?P<tile_v>\d{2})$'
TILENAME_MATCHER = re.compile(TILENAME_REGEX)


def ndvi_hdf_for_tile(tile_name, hdf_dir, satellite=None):
    """
    List the available NDVI HDF files for the given tilename
    Args:
        tile_name: the tile name (e.g. h17v07)
        hdf_dir: directory containing one subdirectory per year which contains
                 HDF files
    Returns:
        list: A list of (full filepath, timestamp_ms) tuples, sorted
              by timestamp_ms
    """
    files = ndvi_list_hdf(hdf_dir, satellite=satellite)
    return files[tile_name]


def parse_tilename(tilename):
    """Turns h10v09 into (10, 9)"""
    m = TILENAME_MATCHER.match(tilename)
    if m is None:
        raise ValueError("Not a valid tilename %s" % tilename)
    h, v = int(m.group('tile_h')), int(m.group('tile_v'))
    return h, v


def parse_ndvi_filename(filepath):
    """
    Extract information (timestamp, tilename) from a MODIS NDVI file
    """
    file_l = filepath.lower()
    m = MODIS_NDVI_MATCHER.match(file_l)
    if m:
        h, v = int(m.group('tile_h')), int(m.group('tile_v'))
        tile_name = 'h%02dv%02d' % (h, v)
        # Not sure if we handle other coll correctly, so fail fast
        assert m.group('coll') == '005'
        # This is a file for our tile
        julian_date = m.group('julian_date')
        date = datetime.strptime(julian_date, '%Y%j')
        timestamp_ms = int(calendar.timegm(date.timetuple()) * 1000.0)
        return {
            'satellite': m.group('sattype'),
            'tile_name': tile_name,
            'h': h,
            'v': v,
            'timestamp_ms': timestamp_ms
        }
    else:
        raise ValueError('Failed to match %s' % filepath)


def ndvi_list_hdf(hdf_dir, satellite=None):
    """
    List all the available HDF files, grouped by tile
    Args:
        hdf_dir: directory containing one subdirectory per year which contains
                 HDF files
        satellite: None to select both Tera and Aqua, 'mod13q1' for MODIS,
                   'myd13q1' for Aqua
    Returns:
        list: A dict (keyed by tilename) of list of (full filepath,
              timestamp_ms) tuples, sorted by timestamp_ms
    """
    files = collections.defaultdict(lambda: [])
    for subdir in os.listdir(hdf_dir):
        subdir = os.path.join(hdf_dir, subdir)
        if not os.path.isdir(subdir):
            continue
        for hdf_file in os.listdir(subdir):
            if not hdf_file.endswith('.hdf'):
                continue
            try:
                full_fname = os.path.join(subdir, hdf_file)
                d = parse_ndvi_filename(hdf_file)
                if satellite is not None and satellite != d['satellite']:
                    continue
                files[d['tile_name']].append((full_fname, d['timestamp_ms']))
            except ValueError as e:
                print e
    for tile_name in files.keys():
        files[tile_name] = sorted(files[tile_name], key=lambda t: t[1])
    return files


def qa_to_qaconf(qa_data):
    return modis_qa.qa_to_qaconf(qa_data)


def qa_to_qaconf_slow(qa_data):
    """
    Transforms MODIS QA into a confidence score.
    confidence comes from the vi_usefulness bits
    """
    # https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod13q1

    # 0-1 MODLAND_QA
    #   00    VI produced, good quality
    #   01    VI produced, but check other QA
    #   10    Pixel produced, but most probably cloudy
    #   11    Pixel not produced due to other reasons than clouds
    # We discards 10 and 11 and check quality for 00 and 01
    vi_quality = (qa_data & 0x3)
    confidence = np.ones_like(qa_data, dtype=np.float32)
    confidence[vi_quality == 3] = 0

    # 2-5 VI usefulness
    #  0000    Highest quality
    #  0001    Lower quality
    #  0010    Decreasing quality
    #  0100    Decreasing quality
    #  1000    Decreasing quality
    #  1001    Decreasing quality
    #  1010    Decreasing quality
    #  1100    Lowest quality
    #  1101    Quality so low that it is not useful
    #  1110    L1B data faulty
    #  1111    Not useful for any other reason/not processed
    vi_usefulness = (qa_data >> 2) & 0xf
    # should assign confidence to points instead
    confidence *= (1. - vi_usefulness / 12.)

    # 6-7 Aerosol quantity
    #  00    Climatology
    #  01    Low
    #  10    Average
    #  11    High
    aerosol = (qa_data >> 6) & 0x3
    confidence[aerosol == 3] = 0

    # 8 Adjacent cloud detected
    #  1    Yes
    #  0    No
    adj_cloud = (qa_data >> 8) & 0x1
    confidence[adj_cloud == 1] = 0

    # 9 Atmosphere BRDF correction performed
    #  1    Yes
    #  0    No
    # TODO: Not sure, but I think we can ignore this. Terra-i does ignore it
    # atm_corr = (qa_data >> 9) & 0x1
    # valid &= atm_corr == 0

    # 10 Mixed Clouds
    #  1    Yes
    #  0    No
    cloud = (qa_data >> 10) & 0x1
    confidence[cloud == 1] = 0

    # 11-13 Land/Water Flag
    #  000    Shallow ocean
    #  001    Land (Nothing else but land)
    #  010    Ocean coastlines and lake shorelines
    #  011    Shallow inland water
    #  100    Ephemeral water
    #  101    Deep inland water
    #  110    Moderate or continental ocean
    #  111    Deep ocean
    # We discard everything but land
    land = (qa_data >> 11) & 0x7
    confidence[land != 1] = 0

    # 14 Possible snow/ice
    #  1    Yes
    #  0    No
    snow = (qa_data >> 14) & 0x1
    confidence[snow == 1] = 0

    # 15 Possible shadow
    #  1    Yes
    #  0    No
    shadow = (qa_data >> 15) & 0x1
    confidence[shadow == 1] = 0

    return confidence


MODIS_NDVI_DATASET_NAME = '250m 16 days NDVI'
MODIS_QA_DATASET_NAME = '250m 16 days VI Quality'

MODIS_NDVI_NODATA = -3000
MODIS_QA_NODATA = 65535


class ModisHDF(object):
    """
    Modis HDF files are basically archive files that contain multiple datasets.
    Typically, they contain NDVI and VI Quality which are the two we're
    interested in.
    """
    def __init__(self, fname):
        self.dataset = gdal.Open(fname)
        assert self.dataset is not None, "Error opening file %s" % fname
        # (url, description) tuples
        self.subdatasets = self.dataset.GetSubDatasets()

    def load_gdal_dataset(self, ds_name):
        for url, description in self.subdatasets:
            if ds_name in description:
                ds = gdal.Open(url)
                return ds
        return None

    def reproject_ndvi_qa(self, reprojector):
        """
        Loads NDVI and QA in WGS84 (terra-i's jGrid projection)
        Returns:
            ndvi
            qa
            mask
        """
        ndvi_ds = self.load_gdal_dataset(MODIS_NDVI_DATASET_NAME)
        ndvi_data = reprojector.reproject(ndvi_ds)

        qa_ds = self.load_gdal_dataset(MODIS_QA_DATASET_NAME)
        qa_data = reprojector.reproject(qa_ds)
        # MODIS stores NDVI as int16
        assert ndvi_data.dtype == np.int16
        # MODIS stores QA as uint16
        assert qa_data.dtype == np.uint16
        # We reinterpret QA as int16 so we have int16 everywhere (doesn't
        # matter for QA since this is a bitmask)
        qa_data = qa_data.view(np.int16)

        # Quick note on NODATA handling :
        #   http://lists.osgeo.org/pipermail/gdal-dev/2013-October/037327.html
        #   http://osgeo-org.1560.x6.nabble.com/gdal-dev-ReprojectImage-and-nodata-td5057270.html
        # Basically, ReprojectImage transforms NODATA into 0. This is kind of
        # a problem because 0 is valid NDVI. On the other hand, it's for
        # rocks, which we do not care about.
        # So we mask everything that has a NDVI equal to 0.
        # We use the same mask for QA because we CANNOT mask QA = 0 since
        # this indicates valid pixels
        mask = ndvi_data != 0

        return ndvi_data, qa_data, mask
