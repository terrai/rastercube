"""
Exports the fractions of a worldgrid to a shapefile.

Example invocation::

    python rastercube/scripts/worldgrid_fracs_to_shapefile
        --grid_root=hdfs:///user/test/ndvi
        --outfile=$HOME/Desktop/ndvi_fracs.shp
"""
import os
import sys
import argparse
import rastercube.jgrid as jgrid
import rastercube.jgrid.utils as jgrid_utils
import rastercube.utils as utils
import rastercube.datasources.shputils as shputils
from osgeo import osr, ogr

parser = argparse.ArgumentParser(description='Create NDVI/QA jgrids from HDF')
parser.add_argument('--grid_root', type=str, required=True,
                    help='grid_root (a fs:// or hdfs://)')
parser.add_argument('--outfile', type=str, required=True,
                    help='output shapefile (e.g. $HOME/Desktop/test.shp)')

if __name__ == '__main__':
    args = parser.parse_args()

    grid_root = args.grid_root
    outfname = args.outfile
    assert outfname.endswith('.shp'), "You should specify a .shp as outfile"

    if os.path.exists(outfname):
        if not utils.confirm(prompt='%s already exists, overwrite ?', resp=False):
            sys.exit(-1)
        os.unlink(outfname)

    header = jgrid.Header.load(grid_root)
    fractions = [f[0] for f in header.list_available_fractions()]

    frac_num = fractions[0]

    print 'Header : ', header
    print 'fractions shape : ', header.frac_height, header.frac_width
    print 'fractions count : ', len(fractions)
    print 'fractions : ', fractions
    print 'outfname : ', outfname

    wgs84_sr = osr.SpatialReference()
    wgs84_sr.ImportFromEPSG(4326)

    polygons = []
    poly_attrs = []
    fields_defs = [('frac_num', ogr.OFTInteger)]
    for frac_num in fractions:
        poly = jgrid_utils.poly_latlng_for_frac(header, frac_num)
        polygons.append(poly)
        poly_attrs.append({'frac_num': frac_num})

    shputils.polygon_to_shapefile(
        polygons, wgs84_sr, outfname, fields_defs, poly_attrs)
    print 'Written to ', outfname
