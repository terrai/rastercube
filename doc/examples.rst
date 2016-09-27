========
Examples
========

Importing a new tile into HDFS
==============================
To import a new tile, you first want to download the HDF files for said tile.

Edit rastercube/config.py
-------------------------
The first step is to modify ``rastercube/config.py`` to add the tile name to
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


Example notebooks
=================

.. toctree::

   notebooks/load_ndvi_glcf.ipynb
   notebooks/load_ndvi_qa.ipynb
