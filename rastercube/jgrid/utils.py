"""
Utility functions related to jGrid2
"""
import gdal
import numpy as np
import numpy.ma as ma
import rastercube.imutils as imutils
import rastercube.gdal_utils as gdal_utils


def fracs_for_poly_bbox_xy(header, polygon_xy):
    """
    Returns fractions covered by the given polygon. This is based on the
    polygon's bounding box.
    """
    assert np.all([header.in_bounds_xy(p) for p in polygon_xy]), \
        "Polygon not contained in jgrid"
    xy_from, xy_to = polygon_xy.min(axis=0), polygon_xy.max(axis=0)
    return header.fracs_for_rect_xy(xy_from, xy_to)


def fracs_for_poly_bbox_latlng(header, polygon_latlng):
    poly_xy = np.array([header.latlng2xy(p) for p in polygon_latlng])
    return fracs_for_poly_bbox_xy(header, poly_xy)


def load_poly_xy_from_jgrid(header, polygon_xy, **kwargs):
    """
    Given a header and a polygon (*assumed* to be fully contained in the
    jgrid), returns a masked array containing the jgrid data in the polygon.

    The returned masked array has the shape of the polygon bounding box but
    only pixels inside the polygon are unmasked
    """
    assert np.all([header.in_bounds_xy(p) for p in polygon_xy]), \
        "Polygon not contained in jgrid"
    xy_from, xy_to = polygon_xy.min(axis=0), polygon_xy.max(axis=0)

    ndvi_data = header.load_slice_xy(xy_from, xy_to, **kwargs)

    poly_mask = imutils.rasterize_poly(polygon_xy - xy_from, ndvi_data.shape)

    return ndvi_data, poly_mask, xy_from


def load_poly_latlng_from_jgrid(header, polygon_latlng, **kwargs):
    """
    Like `load_poly_xy_from_jgrid`, but the polygon is given in latlng
    """
    poly_xy = np.array([header.latlng2xy(p) for p in polygon_latlng])
    return load_poly_xy_from_jgrid(header, poly_xy, **kwargs)


def load_poly_latlng_from_multi_jgrids(headers, polygon, **kwargs):
    """
    Given a set of jgrid header, loads the given polygon from all all grids
    and reproject all of them on the first one.

    Returns:
        xy_from: A single xy_from
        Followed by a list of data/mask pairs :
            data0, mask0, data1, mask1, data2, mask2, ...
    """
    header0 = headers[0]
    data0, mask0, xy_from0 = load_poly_latlng_from_jgrid(header0, polygon,
                                                         **kwargs)

    retval = [xy_from0, data0, mask0]
    for _h in headers[1:]:
        _data, _mask, _xy_from = load_poly_latlng_from_jgrid(_h, polygon,
                                                             **kwargs)
        # only reproject if needed
        if (not _h.spatialref.IsSame(header0.spatialref)) or \
           (_h.geot != header0.geot):
            _data, _mask = reproject_jgrid_on_jgrid(
                header0, xy_from0, data0.shape,
                _h, _xy_from, _data, _mask
            )
        retval.append(_data)
        retval.append(_mask)
    return retval


def poly_latlng_for_frac(header, frac_num):
    """
    Returns the latlng polygon corresponding to a given fraction
    """
    poly = [
        header.xy2latlng((header.x_start(frac_num),
                          header.y_start(frac_num))),
        header.xy2latlng((header.x_end(frac_num),
                          header.y_start(frac_num))),
        header.xy2latlng((header.x_end(frac_num),
                          header.y_end(frac_num))),
        header.xy2latlng((header.x_start(frac_num),
                          header.y_end(frac_num)))
    ]
    return np.array(poly)


def headers_are_same_geogrid(header1, header2):
    """
    Given two headers, verify that they are in the same projection with the
    same geotransform and the same fraction sizes
    """
    return header1.spatialref.IsSame(header2.spatialref) and \
           (header1.geot == header2.geot) and \
           header1.width == header2.width and \
           header1.height == header2.height and \
           header1.frac_width == header2.frac_width and \
           header1.frac_height == header2.frac_height


def load_frac_from_multi_jgrids(headers, frac_num, **kwargs):
    """
    Given a set of jgrid headers and a frac_num in headers[0], loads the
    corresponding area from all headers

    Returns:
        xy_from: A single xy_from
        Followed by a list of data/mask pairs :
            data0, mask0, data1, mask1, data2, mask2, ...
    """
    header0 = headers[0]
    xy_from0 = (header0.x_start(frac_num), header0.y_start(frac_num))
    data0 = header0.load_frac_by_num(frac_num, **kwargs)
    mask0 = np.ones((data0.shape[0], data0.shape[1]), dtype=np.bool)

    frac_poly = poly_latlng_for_frac(header0, frac_num)

    retval = [xy_from0, data0, mask0]
    for _h in headers[1:]:
        if headers_are_same_geogrid(header0, _h):
            print 'Headers in same geogrid'
            _data = _h.load_frac_by_num(frac_num, **kwargs)
            _mask = np.ones((_data.shape[0], _data.shape[1]), dtype=np.bool)
        else:
            _data, _mask, _xy_from = load_poly_latlng_from_jgrid(
                _h, frac_poly, **kwargs)
            _data, _mask = reproject_jgrid_on_jgrid(
                header0, xy_from0, data0.shape,
                _h, _xy_from, _data, _mask
            )

        retval.append(_data)
        retval.append(_mask)

    return retval


def latlng_for_grid(header, xy_from, shape):
    """
    For each point in the grid, computes its latlng coordinates, returning
    a (shape[0], shape[1], 2) array
    """
    yx = np.indices(shape)
    yx[0] += xy_from[1]
    yx[1] += xy_from[0]
    latlng = [header.xy2latlng((x, y))
              for y, x in zip(yx[0].reshape(-1), yx[1].reshape(-1))]
    return np.array(latlng).reshape(shape[0], shape[1], 2)


def slice_and_reproject_to_grid(header, xy_from, grid_shape, src_ds,
                                interpolation='near'):
    """
    Helper function which takes a jgrid slice (so Header, xy_from, grid_shape)
    and a GDAL dataset and slice/reprojects the GDAL dataset to the jgrid
    slice.

    This is typically useful to reproject some arbitrary TIFF file on some
    part of the NDVI worldgrid.

    Args:
        header: A jgrid3.Header
        xy_from: the (x, y) at which the subgrid starts in the given header
        grid_shape: the (height, width) of the subgrid
        src_ds: The source GDAL dataset to reproject
        interpolation: The resampling mode : one of 'near', 'mode', 'average'

    Returns:
        A masked array containing the reprojected values
    """
    # https://jgomezdans.github.io/gdal_notes/reprojection.html
    # http://www.gdal.org/gdalwarper_8h.html#ad36462e8d5d34642df7f9ea1cfc2fec4
    src_wkt = src_ds.GetProjectionRef()

    nbands = src_ds.RasterCount
    src_dtype = src_ds.GetRasterBand(1).DataType
    # print 'src dtype : %s' % gdal.GetDataTypeName(src_dtype)

    mem_drv = gdal.GetDriverByName('MEM')
    dst_ds = mem_drv.Create('', grid_shape[1], grid_shape[0], nbands,
                            src_dtype)
    dst_geo = header.geot_for_xyfrom(xy_from)
    dst_ds.SetGeoTransform(dst_geo)
    dst_ds.SetProjection(header.spatialref.ExportToWkt())

    # NoData handling when using ReprojectImage with a MEM target ds is
    # a bit tricky. See those discussions :
    # https://trac.osgeo.org/gdal/ticket/6404
    # http://gis.stackexchange.com/q/158503
    # We have to fill each band with the nodata value before doing the
    # reprojectimage because the bands are initialized with 0
    ndv = None
    for i in range(1, nbands + 1):
        src_b = src_ds.GetRasterBand(i)
        if ndv is not None and not np.isnan(ndv):
            assert src_b.GetNoDataValue() == ndv, \
                "All bands of the source dataset should have the same NODATA"
        else:
            ndv = src_b.GetNoDataValue()
        dst_b = dst_ds.GetRasterBand(i)
        if ndv is not None:
            dst_b.SetNoDataValue(ndv)
            dst_b.Fill(ndv)

    if interpolation == 'near':
        gdal_mode = gdal.GRA_NearestNeighbour
    elif interpolation == 'mode':
        gdal_mode = gdal.GRA_Mode
    elif interpolation == 'average':
        gdal_mode = gdal.GRA_Average
    else:
        raise ValueError("Invalid interpolation mode %s" % interpolation)

    res = gdal.ReprojectImage(
        src_ds,
        dst_ds,
        src_ds.GetProjectionRef(),
        dst_ds.GetProjectionRef(),
        gdal_mode
    )
    assert res == 0, 'Error reprojecting, res=%d' % res
    dst_arr = dst_ds.ReadAsArray()

    # GDAL ReadAsArray returns (bands, height, width) but we expect
    # (height, width, bands)
    if len(dst_arr.shape) == 3:
        dst_arr = dst_arr.transpose(1, 2, 0)

    # TODO: This assumes that the no data value is the same for all bands
    if ndv is not None:
        dst_arr = ma.masked_where(dst_arr == ndv, dst_arr)
    else:
        dst_arr = ma.asarray(dst_arr)

    return dst_arr


def gdal_ds_from_jgrid_slice(header, xy_from, data):
    """
    Returns a GDAL in-memory dataset that maps the provided jgrid slice.
    Note that the dataset only keeps a reference to the data array.
    """
    ds = gdal_utils.gdal_ds_from_array(data)
    ds.SetGeoTransform(header.geot_for_xyfrom(xy_from))
    ds.SetProjection(header.spatialref.ExportToWkt())
    return ds


def reproject_jgrid_on_jgrid(target_header, target_xy_from, target_shape,
                             src_header, src_xy_from, src_data, src_mask):
    """
    Reproject a source jgrid on a target jgrid
    """
    data_ds = gdal_ds_from_jgrid_slice(src_header, src_xy_from, src_data)
    # This requires a mask copy because GDAL doesn't support bool
    # Also, GDAL ignores 0 during reproject, so add 1 to the mask here
    src_mask = src_mask.astype(np.uint8) + 1
    mask_ds = gdal_ds_from_jgrid_slice(src_header, src_xy_from, src_mask)

    new_data = slice_and_reproject_to_grid(target_header, target_xy_from,
                                           target_shape, data_ds)
    new_mask = slice_and_reproject_to_grid(target_header, target_xy_from,
                                           target_shape, mask_ds)
    # recover the boolean mask
    new_mask = new_mask > 1

    return new_data, new_mask
