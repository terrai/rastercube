"""
GDAL-related utility functions
"""
import gdal
import numpy as np
from osgeo import osr, gdal_array


def latlng_bounding_box_from_ds(ds):
    """
    Given a GDAL dataset, computes its bounding box int lat/lng
    """
    wgs84_sr = osr.SpatialReference()
    wgs84_sr.ImportFromEPSG(4326)

    ds_sr = osr.SpatialReference()
    ds_sr.ImportFromWkt(ds.GetProjection())

    ds_to_wgs84 = osr.CoordinateTransformation(ds_sr, wgs84_sr)

    geot = ds.GetGeoTransform()
    def _xy2latlng(x, y):
        x_geo = geot[0] + x * geot[1] + y * geot[2]
        y_geo = geot[3] + x * geot[4] + y * geot[5]
        lng, lat, _ = ds_to_wgs84.TransformPoint(x_geo, y_geo)
        return lat, lng
    poly = [
        _xy2latlng(0, 0),
        _xy2latlng(ds.RasterXSize, 0),
        _xy2latlng(ds.RasterXSize, ds.RasterYSize),
        _xy2latlng(0, ds.RasterYSize)
    ]
    return poly


def gdal_ds_from_array(arr):
    """
    Returns an in-memory GDAL dataset that is a proxy to the given array
    http://www.gdal.org/frmt_mem.html
    """
    # Ensure 3D
    arr = arr.reshape(arr.shape[0], arr.shape[1], -1)

    assert arr.flags.c_contiguous
    # http://stackoverflow.com/questions/11264838/how-to-get-the-memory-address-of-a-numpy-array-for-c
    pointer, read_only_flag = arr.__array_interface__['data']
    gdal_dtype = gdal_array.NumericTypeCodeToGDALTypeCode(arr.dtype)
    print "arr dtype : ", arr.dtype, " => gdal dtype : ", \
        gdal.GetDataTypeName(gdal_dtype)
    ds = gdal.Open("MEM:::" + ",".join([
        "DATAPOINTER=%d" % pointer,
        "DATATYPE=%d" % gdal_dtype,
        "LINES=%d" % arr.shape[0],
        "PIXELS=%d" % arr.shape[1],
        "BANDS=%d" % arr.shape[2],
        "LINEOFFSET=%d" % arr.strides[0],
        "PIXELOFFSET=%d" % arr.strides[1],
        "BANDOFFSET=%d" % arr.strides[2]
    ]))
    assert ds is not None, "Failed to create in-memory dataset"
    # Kind of ugly hack to keep a reference on arr to avoid garbage collection
    ds._arr = arr
    return ds


def gdal_warp(src_ds, dst_wkt):
    """
    GDAL warp but in python

    Args:
        src_ds: Source dataset
        dst_srs: Target SRS WKT

    Returns:
        dst_ds: The warped dataset

    http://gis.stackexchange.com/a/140053
    """
    error_threshold = 0.125  # error threshold --> use same value as in gdalwarp
    resampling = gdal.GRA_NearestNeighbour

    # Call AutoCreateWarpedVRT() to fetch default values for target
    # raster dimensions and geotransform
    tmp_ds = gdal.AutoCreateWarpedVRT(src_ds,
                                      None, # src_wkt from source
                                      dst_wkt,
                                      resampling,
                                      error_threshold)
    dst_ds = gdal.GetDriverByName('MEM').CreateCopy('', tmp_ds)

    return dst_ds


def latlng_bbox_from_ds(ds):
    wgs84_sr = osr.SpatialReference()
    wgs84_sr.ImportFromEPSG(4326)

    ds_sr = osr.SpatialReference()
    ds_sr.ImportFromWkt(ds.GetProjectionRef())

    ds_to_wgs84 = osr.CoordinateTransformation(
            ds_sr, wgs84_sr)

    gt = ds.GetGeoTransform()

    minx = gt[0]
    miny = gt[3] + ds.RasterXSize * gt[4] + ds.RasterYSize * gt[5]
    maxx = gt[0] + ds.RasterXSize * gt[1] + ds.RasterYSize * gt[2]
    maxy = gt[3]

    def t(x, y):
        lng, lat, _ = ds_to_wgs84.TransformPoint(x, y)
        return lat, lng

    return np.array([
        t(minx, miny),
        t(maxx, miny),
        t(maxx, maxy),
        t(minx, maxy)
    ])
