"""
This defines world grids that are used to index jGridFrac
"""
import numpy as np
from osgeo import osr


class GLCFGrid(object):
    """
    This is a helper class to import Global Land Cover Facilitydata.
    GLCF is using the WGS84
    """
    # GLCF data description
    # http://glcf.umd.edu/data/lc/

    # GLCF tiles each cover 4 UTM zones (tilename is VU3334, meaning it coves
    # UTM V33 V34, U33, U34)
    # TODO: That is incorrect for rows X and C
    GLCF_tile_height = 3840
    GLCF_tile_width = 2880

    # UTM has 60 horizontal tiles and 20 vertical (C to X)
    # Since each GLCF tile covers 4 UTM tiles, we divide by 2 on each axis
    GLCF_n_tiles_x = 30
    GLCF_n_tiles_y = 10

    # There is no temporal data in GLCF, so we can use large cells
    # 3840 x 2880 gives us ~9MB cells (3840x2880 / (1024x1024))
    # cell_width and cell_height MUST divide GLCF_tile_height / GLCF_tile_width
    cell_width = GLCF_tile_width
    cell_height = GLCF_tile_height

    n_cells_per_tile_x = GLCF_tile_width / cell_width
    n_cells_per_tile_y = GLCF_tile_height / cell_height

    n_cells_x = (GLCF_tile_width * GLCF_n_tiles_x) / cell_width
    n_cells_y = (GLCF_tile_height * GLCF_n_tiles_y) / cell_height

    width = GLCF_n_tiles_x * GLCF_tile_width
    height = GLCF_n_tiles_y * GLCF_tile_height

    # The geotransform defines the origin of the grid (tile 0,0) relative to
    # the origin of the projection, which is greenwich meridian (so we have
    # to subtract half of the grid)
    # The geotransform maps pixel coordinates into georeferenced coords
    # with
    #      Xgeo = GT(0) + Xpixel*GT(1) + Yline*GT(2)
    #      Ygeo = GT(3) + Xpixel*GT(4) + Yline*GT(5)
    #
    # Obtained from gdalinfo
    pix_size = (0.004166666666667, -0.004166666666667)
    geo_orig = (-GLCF_tile_width * pix_size[0] * GLCF_n_tiles_x / 2,
                -GLCF_tile_height * pix_size[1] * GLCF_n_tiles_y / 2)
    geot = (geo_orig[0], pix_size[0], 0.0,
            geo_orig[1], 0.0, pix_size[1])

    # Obtained by runing
    # gdalinfo "MCD12Q1_V51_LC1.2004.FE1718.tif"
    proj_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433],
        AUTHORITY["EPSG","4326"]]
    """

    # Not all letters correspond to row, so store a mapping here
    ROW_MAP = {
        'X': 0, 'W': 1, 'V': 2, 'U': 3, 'T': 4, 'S': 5, 'R': 6, 'Q': 7, 'P': 8,
        'N': 9, 'M': 10, 'L': 11, 'K': 12, 'J': 13, 'H': 14, 'G': 15, 'F': 16,
        'E': 17, 'D': 18, 'C': 19
    }

    # Our fractions are numbered starting at (0,0) on the GLCF grid and
    # then following each row
    # Since we have a one-to-one mapping between tile and fractions, it's
    # pretty easy to convert
    def get_cell_for_tile(self, tile):
        """
        Returns the cell corresponding to the given GLCF tile. The tile
        is identified by its UTM identifier (e.g. HG5152)
        """
        assert self.cell_width == self.GLCF_tile_width
        assert self.cell_height == self.GLCF_tile_height
        row_from = tile[0]
        row_to = tile[1]
        col_from = int(tile[2:4])
        col_to = int(tile[4:6])

        i = self.ROW_MAP[row_from.upper()] / 2
        j = (col_from - 1) / 2
        # print "GLCF cell ", i, j

        fracnum = np.ravel_multi_index((i, j), (self.n_cells_y, self.n_cells_x))
        return fracnum


class MODISGrid(object):
    """
    This is a helper class to import MODIS data into a jGrid that uses the
    MODIS sinusoidal projection.
    """
    # MODIS tiles ALL have a resolution of 4800x4800
    # https://lpdaac.usgs.gov/dataset_discovery/modis/modis_products_table/mod13q1
    MODIS_tile_height = 4800
    MODIS_tile_width = 4800

    # The MODIS grid has 36 horizontal tile number and 18 vertical
    # http://modis-land.gsfc.nasa.gov/MODLAND_grid.html
    MODIS_n_tiles_x = 36
    MODIS_n_tiles_y = 18

    #  We should aim  for a fraction size of 128MB for float fractions
    # (containing all the 351 times we have) since that's the size of a
    # hadoop block. If we aim for 600 dates (so we can add data up to 2025
    # before we have to use 2 blocks for a frac), we should aim for 200x200
    #   200x200x600x4 / (1024x1024) = 91 MB
    # But since this is compressed, we can increase the size
    cell_width = 400  # this is ASSUMED to divide 4800
    cell_height = 400

    n_cells_per_tile_x = MODIS_tile_width / cell_width
    n_cells_per_tile_y = MODIS_tile_height / cell_height

    n_cells_x = (MODIS_tile_width * MODIS_n_tiles_x) / cell_width
    n_cells_y = (MODIS_tile_height * MODIS_n_tiles_y) / cell_height

    width = MODIS_n_tiles_x * MODIS_tile_width
    height = MODIS_n_tiles_y * MODIS_tile_height

    # Here the geotransform only applies pixel size scaling. It also defines
    # the origin of the grid (tile 0,0) relative to the origin of the
    # projection, which is greenwich meridian (so we have to subtract half
    # of the grid)
    pix_size = (231.65635826374995, -231.65635826395834)
    geo_orig = (-MODIS_tile_width * pix_size[0] * MODIS_n_tiles_x / 2,
                -MODIS_tile_height * pix_size[1] * MODIS_n_tiles_y / 2)
    geot = (geo_orig[0], pix_size[0], 0.0,
            geo_orig[1], 0.0, pix_size[1])

    # Obtained by runing
    # gdalinfo "MOD13Q1.A2000049.h27v06.005....hdf" -sd 1
    # This is this sr-org:6842
    # http://spatialreference.org/ref/sr-org/6842/
    proj_wkt = """
    PROJCS["unnamed",
        GEOGCS["Unknown datum based upon the custom spheroid",
            DATUM["Not specified (based on custom spheroid)",
                SPHEROID["Custom spheroid",6371007.181,0]],
            PRIMEM["Greenwich",0],
            UNIT["degree",0.0174532925199433]],
        PROJECTION["Sinusoidal"],
        PARAMETER["longitude_of_center",0],
        PARAMETER["false_easting",0],
        PARAMETER["false_northing",0],
        UNIT["Meter",1]]
    """

    # Our fractions are numbered starting at (0,0) on the MODIS grid and
    # then following each row
    def get_cells_for_tile(self, tile_h, tile_v):
        """
        Returns the list of cells covered by the given modis tile. The tile
        is identified by its MODIS grid coordinates
        """
        range_x = np.arange(tile_h * self.n_cells_per_tile_x,
                            (tile_h + 1) * self.n_cells_per_tile_x)
        range_y = np.arange(tile_v * self.n_cells_per_tile_y,
                            (tile_v + 1) * self.n_cells_per_tile_y)
        cells_ij = np.dstack(
            np.meshgrid(range_y, range_x, indexing='ij')).reshape(-1, 2)
        cells = np.ravel_multi_index(
            (cells_ij[:, 0], cells_ij[:, 1]),
            (self.n_cells_y, self.n_cells_x)
        )
        # sanity check
        assert len(cells) == self.n_cells_per_tile_x * self.n_cells_per_tile_y
        return cells

    def tile_xy_from(self, tile_h, tile_v):
        return (tile_h * self.MODIS_tile_width,
                tile_v * self.MODIS_tile_height)

    def tile_for_cell(self, cell):
        cell_i, cell_j = np.unravel_index(
            cell, (self.n_cells_y, self.n_cells_x))
        h = int(cell_j / self.n_cells_per_tile_x)
        v = int(cell_i / self.n_cells_per_tile_y)
        return h, v

    def get_cell_indices_in_tile(self, cell, tile_h, tile_v):
        """
        Given a particular cell and the tile that contains it, returns
        the view ((i_from, i_to), (j_from, j_to)) covered by the cell
        relative to the tile. i_to and j_to are exclusive
        So this means
          cell_data = tile_data[i_from:i_to, j_from:j_to]
        """
        assert self.tile_for_cell(cell) == (tile_h, tile_v), \
            "Provided cell is  not covered by provided tile"
        tile_x_range = (tile_h * self.MODIS_tile_width,
                        (tile_h + 1) * self.MODIS_tile_width)
        tile_y_range = (tile_v * self.MODIS_tile_height,
                        (tile_v + 1) * self.MODIS_tile_height)
        cell_i, cell_j = np.unravel_index(
            cell, (self.n_cells_y, self.n_cells_x))
        cell_x_range = (cell_j * self.cell_width,
                        (cell_j + 1) * self.cell_width)
        cell_y_range = (cell_i * self.cell_height,
                        (cell_i + 1) * self.cell_height)
        # TODO: assert that cell_x_range is contained in tile_x_range and
        # same for y
        return ((cell_y_range[0] - tile_y_range[0],
                 cell_y_range[1] - tile_y_range[0]),
                (cell_x_range[0] - tile_x_range[0],
                 cell_x_range[1] - tile_x_range[0]))
