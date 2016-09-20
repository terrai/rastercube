import numpy as np
import unittest
from rastercube.worldgrid.grids import MODISGrid
from numpy.testing import assert_array_equal


class MODISGridTest(unittest.TestCase):
    def test_tile_for_cell(self):
        """
        Test coherence of get_cells_for_tile and tile_for_cell
        """
        h, v = (10, 8)
        grid = MODISGrid()
        cells = grid.get_cells_for_tile(h, v)
        for cell in cells:
            self.assertEquals(grid.tile_for_cell(cell), (h, v))

    def test_cell_indices_in_tile(self):
        """
        Test get_cell_indices_in_tile by filling an int array for a tile,
        using the indices returned by cell_indices_in_tile for each cell
        in the tile. The array should be fully filled with 1 at the end
        """
        h, v = (20, 11)
        grid = MODISGrid()
        tile_data = np.zeros(
            (MODISGrid.MODIS_tile_height, MODISGrid.MODIS_tile_width),
            dtype=np.int16)
        cells = grid.get_cells_for_tile(h, v)
        for cell in cells:
            i_range, j_range = grid.get_cell_indices_in_tile(cell, h, v)
            tile_data[i_range[0]:i_range[1], j_range[0]:j_range[1]] += 1
        # If tile_data contains some zeros, this means the tile is not
        # fully covered by the cells. If it contains values > 1, this means
        # than more than one cell covers a given tile pixel
        assert_array_equal(tile_data, np.ones_like(tile_data))
