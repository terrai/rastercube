from osgeo import gdal
import numpy as np


def compute_geo_transform_from_jgrid_meta(meta):
    """
    Computes a 6-tuple Geo Transform from a jGrid meta.
    """
    pix_xsize = meta['cellsize']
    pix_ysize = -meta['cellsize']
    offset_x = offset_y = 0
    lng_tl = meta['top_left_lng']
    lat_tl = meta['top_left_lat']
    return (lng_tl, pix_xsize, offset_x, lat_tl, offset_y, pix_ysize)


def compute_geo_transform_from_jgrid2_header(header):
    """
    Computes a 6-tuple Geo Transform from a jGrid2 Header.
    """
    lat_tl, lng_tl = header.tl_latlng
    pix_xsize = header.cell_size
    pix_ysize = -pix_xsize
    offset_x = offset_y = 0
    return (lng_tl, pix_xsize, offset_x, lat_tl, offset_y, pix_ysize)


def write_terrai_zip_to_tiff(name, data, meta, projection, unit):
    """
    Writes a numpy array from a terrai zip to a tiff file given the
    corresponding jGrid meta and the ouptut gdal projection and unit.
    Keyword arguments:
    name -- the name of the output file
    data -- the numpy array that will be written
    meta -- the jGrid meta for the numpy array
    projection -- the gdal projection to use for the output file
    """
    gtiff_drv = gdal.GetDriverByName("GTiff")
    tiff_file = gtiff_drv.Create(name, data.shape[1], data.shape[0], 1, unit)

    tiff_file.SetGeoTransform(compute_geo_transform_from_jgrid_meta(meta))
    tiff_file.SetProjection(projection)

    tiff_file.GetRasterBand(1).WriteArray(data)
    tiff_file.FlushCache()
    tiff_file = None


def reproject_tiff_from_model(old_name, new_name, model, unit):
    """
    Reprojects an tiff on a tiff model. Can be used to warp tiff.
    Keyword arguments:
    old_name -- the name of the old tiff file
    new_name -- the name of the output tiff file
    model -- the gdal dataset which will be used to warp the tiff
    unit -- the gdal unit in which the operation will be performed
    """
    mem_drv = gdal.GetDriverByName("MEM")

    old = gdal.Open(old_name)

    new = mem_drv.Create(new_name, model.RasterXSize, model.RasterYSize, 1,
                         unit)

    new.SetGeoTransform(model.GetGeoTransform())
    new.SetProjection(model.GetProjection())

    res = gdal.ReprojectImage(old, new, old.GetProjection(),
                              model.GetProjection(), gdal.GRA_NearestNeighbour)

    assert res == 0, 'Error in ReprojectImage'
    arr = new.ReadAsArray()
    new = None
    return arr


def reproject_tiff_custom(old_name, new_name, new_x_size, new_y_size,
                          new_geo_transform, new_projection, unit, mode):
    """
    Reprojects an tiff with custom tiff arguments. Can be used to warp tiff.
    If no projection is provided, fallback to old projection.
    Keyword arguments:
    old_name -- the name of the old tiff file
    new_name -- the name of the output tiff file
    new_x_size -- the number of new size in x dimension
    new_y_size -- the number of new size in y dimension
    new_geo_transform -- the new geo transform to apply
    new_projection -- the new projection to use
    unit -- the gdal unit in which the operation will be performed
    mode -- the gdal mode used for warping
    """
    mem_drv = gdal.GetDriverByName("MEM")

    old = gdal.Open(old_name)
    r = old.GetRasterBand(1)
    r.GetNoDataValue()

    # Adds 1 to keep the original zeros (reprojectImage maps NO_DATA to 0)
    old_array = old.ReadAsArray()
    mask = old_array == old.GetRasterBand(1).GetNoDataValue()
    old_array += 1
    old_array[mask] = 0
    temp = mem_drv.Create("temp", old.RasterXSize, old.RasterYSize, 1, unit)
    temp.SetGeoTransform(old.GetGeoTransform())
    temp.SetProjection(old.GetProjection())
    temp.GetRasterBand(1).WriteArray(old_array)

    new = mem_drv.Create(new_name, new_x_size, new_y_size, 1, unit)

    new.SetGeoTransform(new_geo_transform)
    if new_projection is None:
        new.SetProjection(old.GetProjection())
    else:
        new.SetProjection(new_projection)

    res = gdal.ReprojectImage(temp, new, old.GetProjection(), new_projection,
                              mode)

    assert res == 0, 'Error in ReprojectImage'
    arr = new.ReadAsArray()
    mask = arr != 0
    arr -= 1
    arr[~mask] = 0
    new = None
    temp = None

    return arr, mask


def lat_lon_to_pixel(transform, lat_lon):
    x = (lat_lon[1] - transform[0]) / transform[1]
    y = (lat_lon[0] - transform[3]) / transform[5]
    return (int(y), int(x))


def write_int16_to_tiff(name, data, sr, geot, nodata_val=None):
    assert data.dtype == np.int16
    gtiff_drv = gdal.GetDriverByName("GTiff")
    tiff_file = gtiff_drv.Create(name, data.shape[1], data.shape[0], 1,
                                 gdal.GDT_Int16,
                                 options=['COMPRESS=DEFLATE', 'ZLEVEL=1'])
    tiff_file.SetGeoTransform(geot)
    tiff_file.SetProjection(sr)

    band = tiff_file.GetRasterBand(1)
    if nodata_val is not None:
        band.SetNoDataValue(nodata_val)
    band.WriteArray(data)
    band.FlushCache()
    del band
    del tiff_file


def write_int16_to_tiff_from_header(name, data, header, xy_from,
                                    nodata_val=None):
    """
    Given an int16 data array and a (header, xy_from), writes the data to
    a tiff
    """
    write_int16_to_tiff(
        name,
        data,
        header.spatialref.ExportToWkt(),
        header.geot_for_xyfrom(xy_from),
        nodata_val
    )
