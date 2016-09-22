========
Overview
========

`rastercube` is a python package to store large geographical raster collections
on the Hadoop File System (HDFS). The initial use case was to store and quickly
access MODIS NDVI timeseries for any pixel on the whole planet.

In addition, `rastercube` provides facility to process the data using Spark.

All the geographical computations are handled by `gdal`.

Worldgrid
=========
A worldgrid is a (chunked) georeferenced nD (n >= 2) array covering the whole
earth. The first two dimensions are spatial dimensions and the third dimension
is usually time.

The original use case for which the format was designed was to store MODIS
NDVI timeseries for the pantropical areas (south america, africa and asia).
MODIS NDVI images have a frequency of 8 days (if you use Terra and Aqua) and
a ground resolution of 250x250m.

In the case of MODIS NDVI, there is also a quality indicator. So you would
have two worldgrids : one containing NDVI and one containing quality. You could
imagine more worldgrids if you were interested in other MODIS bands. You can
also have worldgrid for other data types such as Global Land Cover data.

The rastercube package provides a simple API to store worldgrids on the Hadoop
File System

Date management
===============
Geographical rasters are usually downloaded as timestamped tiles. When
creating a worldgrid, you want to ensure that you have the same dates available
for each tile so you have a homogeneous time axis for the whole grid.

This is done by collecting the available date for a particular model tile and
storing them in a .csv. See the ``ndvi_collect_dates.py`` script.


.. note::

    If a particular MODIS tile is missing a date (this can happen due to
    satellite malfunction), you can just manually create a tile with all
    bad QA/NDVI

Input data files structure
==========================
The input data (e.g. the MODIS HDF tiles, the GLCF tiles) that are imported
into worldgrid need to be stored on a traditional unix filesystem (or on NFS).

The environment variable ``$RASTERCUBE_DATA`` should point to the location
where the data is stored.

For example, here is the layout used in the terra-i project. `rastercube`
assumes that you have similar ``0_input`` and ``1_manual`` directories.

- ``0_input`` should contain all input raster data
- ``1_manual`` should contain some manually created files like the NDVI dates
  .csv

::

    /home/julien/terra_i/sv2454_data
    ├── 0_input
    │   ├── glcf_5.1
    │   ├── hansen_dets
    │   ├── landsat8
    │   ├── MODIS_HDF
    │   ├── modis_www_mirror
    │   ├── prm
    │   ├── terrai_dets
    │   └── TRMM_daily
    ├── 1_manual
    │   ├── hansen_wms
    │   ├── ndvi_dates.csv
    │   ├── ndvi_dates.short.csv
    │   ├── ndvi_dates.terra_aqua.csv
    │   ├── ndvi_dates.terra_aqua.short.csv
    │   ├── ndvi_dates.terra.csv
    │   └── qgis
    ├── 2_intermediate
    │   ├── logs
    │   ├── models
    │   ├── models_rawndvi
    │   └── terrai_dets
    ├── experimental
    │   ├── glcf_dets
    │   ├── glcf_dets_2
    │   ├── land_cover
    │   ├── qgis
    │   ├── test2.hdf
    │   └── test.hdf

