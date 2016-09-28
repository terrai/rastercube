========
Examples
========

.. warning::

    Ensure you have read the :ref:`configuration` and :ref:`envvar` sections

Importing a new tile into HDFS
==============================
To import a new tile, you first want to download the HDF files for said tile.

Edit your configuration file
----------------------------
The first step is to modify your :ref:`configuration` file to add the tile name to
``MODIS_TERRA_TILES``.

For example::

  MODIS_TERRA_TILES = [
      ...
      # central america
      h09v07,
      ...

Run ndvi_hdf_download.py
------------------------
Now, run ``ndvi_hdf_download.py`` which will update the local cache of the
MODIS tile listing and download the tabs specified in the config.

::

  python rastercube/scripts/ndvi_hdf_download.py

Run create_ndvi_worldgrid.py
----------------------------

::

  python rastercube/scripts/create_ndvi_worldgrid.py
      --tile=h09v07
      --worldgrid=hdfs:///user/terrai/worldgrid/
      --dates_csv=$TERRAI_DATA/1_manual/ndvi_dates.terra_aqua.csv


Updating the worldgrid when new dates are available
===================================================

First, update your dates CSV::

  python rastercube/scripts/ndvi_collect_dates.py
    --tile=h09v07
    --outfile=$TERRAI_DATA/1_manual/ndvi_dates.terra_aqua.csv


And then, for each tile, run::

  python rastercube/scripts/complete_ndvi_worldgrid.py
      --tile=h09v07
      --worldgrid=hdfs:///user/terrai/worldgrid/
      --dates_csv=$TERRAI_DATA/1_manual/ndvi_dates.terra_aqua.csv


Example notebooks
=================

.. toctree::

   notebooks/load_ndvi_glcf.ipynb
   notebooks/load_ndvi_qa.ipynb
