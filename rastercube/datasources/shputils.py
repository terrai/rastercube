"""
Utilities to deal with shapefiles
"""
import os
import numpy as np
import gdal
import tempfile
from osgeo import osr, ogr


def load_polygons_from_shapefile(filename, target_sr):
    """
    Loads the given shapefiles, reprojects the polygons it contains in the
    given target spatialreference.
    Returns:
        polygons: A list of polygons (list of points)
        attributes: A list of attributes (dict of strings)
    """
    shape = ogr.Open(filename)
    assert shape, "Couldn't open %s" % filename
    assert shape.GetLayerCount() == 1
    layer = shape.GetLayer()
    nfeatures = layer.GetFeatureCount()

    shape_sr = osr.SpatialReference()
    shape_sr.ImportFromWkt(layer.GetSpatialRef().ExportToWkt())

    # Transform from shape to image coordinates
    transform = osr.CoordinateTransformation(shape_sr, target_sr)

    # http://geoinformaticstutorial.blogspot.ch/2012/10/accessing-vertices-from-polygon-with.html
    polygons = []
    attributes = []
    for i in xrange(nfeatures):
        feature = layer.GetFeature(i)
        attr = feature.items()
        newattr = {}
        # some attributes contains unicode character. ASCIIfy everything
        # TODO: A bit brutal, but easy...
        for k, v in attr.items():
            newk = k.decode('ascii', errors='ignore')
            newattr[newk] = v
        attr = newattr

        geometry = feature.GetGeometryRef()
        assert geometry.GetGeometryName() == 'POLYGON'
        # A ring is a polygon in shapefiles
        ring = geometry.GetGeometryRef(0)
        assert ring.GetGeometryName() == 'LINEARRING'
        # The ring duplicates the last point, so for the polygon to be closed,
        # last point should equal first point
        npoints = ring.GetPointCount()
        points = [ring.GetPoint(i) for i in xrange(npoints)]
        points = [transform.TransformPoint(*p) for p in points]
        points = np.array(points)[:, :2]  # third column is elevation - discard
        # swap (lng, lat) to (lat, lng)
        points = points[:, ::-1]
        assert np.allclose(points[-1], points[0])
        polygons.append(points)
        attributes.append(attr)

    #print len(polygons), ' loaded from ', filename
    return polygons, attributes


def polygon_to_shapefile(polygons, poly_sr, fname, fields_defs=None,
                         poly_attrs=None):
    """
    Write a set of polygons to a shapefile
    Args:
        polygons: a list of (lat, lng) tuples
        fields: The list of fields for those polygons, as a tuple
                (name, ogr type) for each field. For example :
                    [('field1', ogr.OFTReal), ('field2', ogr.OFTInteger)]
        poly_attrs: A list of dict containing the attributes for each polygon
                        [{'field1' : 1.0, 'field2': 42},
                         {'field1' : 3.0, 'field2': 60}]
    """
    shp_driver = ogr.GetDriverByName("ESRI Shapefile")
    out_ds = shp_driver.CreateDataSource(fname)
    assert out_ds is not None, "Failed to create temporary %s" % fname
    out_layer = out_ds.CreateLayer(fname, poly_sr, geom_type=ogr.wkbPolygon)

    has_attrs = fields_defs is not None
    if has_attrs:
        attrs_name = []
        for field_name, field_type in fields_defs:
            out_layer.CreateField(ogr.FieldDefn(field_name, field_type))
            attrs_name.append(field_name)

    layer_defn = out_layer.GetLayerDefn()
    for i, poly in enumerate(polygons):
        ring = ogr.Geometry(ogr.wkbLinearRing)
        # gdal uses the (x, y) convention => (lng, lat)
        for point in poly:
            ring.AddPoint(point[1], point[0])
        ring.AddPoint(poly[0][1], poly[0][0])  # re-add the start to close
        p = ogr.Geometry(ogr.wkbPolygon)
        p.AddGeometry(ring)

        out_feature = ogr.Feature(layer_defn)
        out_feature.SetGeometry(p)

        if has_attrs:
            attrs = poly_attrs[i]
            for field_name in attrs_name:
                out_feature.SetField(field_name, attrs[field_name])

        out_layer.CreateFeature(out_feature)

    out_feature.Destroy()
    out_ds.Destroy()


def rasterize_polygon_like(polygon, poly_sr, model_raster_fname, dtype,
                           options, nodata_val=0):
    """
    Like rasterize_shapefile_like, but for an in-memory polygon
    Args:
        polygons: a list of (lat, lng) tuples
    """
    _, tmp_fname = tempfile.mkstemp(suffix='.shp')
    # mkstemp creates the file but GDAL wants to create it itself
    os.unlink(tmp_fname)
    polygon_to_shapefile([polygon], poly_sr, tmp_fname)

    data = rasterize_shapefile_like(tmp_fname, model_raster_fname, dtype,
                                    options, nodata_val)
    os.unlink(tmp_fname)

    return data


def rasterize_shapefile_like(shpfile, model_raster_fname, dtype, options,
                             nodata_val=0,):
    """
    Given a shapefile, rasterizes it so it has
    the exact same extent as the given model_raster

    `dtype` is a gdal type like gdal.GDT_Byte
    `options` should be a list that will be passed to GDALRasterizeLayers
        papszOptions, like ["ATTRIBUTE=vegetation","ALL_TOUCHED=TRUE"]
    """
    model_dataset = gdal.Open(model_raster_fname)
    shape_dataset = ogr.Open(shpfile)
    shape_layer = shape_dataset.GetLayer()
    mem_drv = gdal.GetDriverByName('MEM')
    mem_raster = mem_drv.Create(
        '',
        model_dataset.RasterXSize,
        model_dataset.RasterYSize,
        1,
        dtype
    )
    mem_raster.SetProjection(model_dataset.GetProjection())
    mem_raster.SetGeoTransform(model_dataset.GetGeoTransform())
    mem_band = mem_raster.GetRasterBand(1)
    mem_band.Fill(nodata_val)
    mem_band.SetNoDataValue(nodata_val)

    # http://gdal.org/gdal__alg_8h.html#adfe5e5d287d6c184aab03acbfa567cb1
    # http://gis.stackexchange.com/questions/31568/gdal-rasterizelayer-doesnt-burn-all-polygons-to-raster
    err = gdal.RasterizeLayer(
        mem_raster,
        [1],
        shape_layer,
        None,
        None,
        [1],
        options
    )
    assert err == gdal.CE_None
    return mem_raster.ReadAsArray()
