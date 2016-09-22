"""
Print infos about a worldgrid

Example invocation::

    python rastercube/scripts/worldgrid_info.py
        --grid_root=hdfs:///user/test/ndvi
"""
import argparse
import rastercube.jgrid as jgrid

parser = argparse.ArgumentParser(description='Create NDVI/QA jgrids from HDF')
parser.add_argument('--grid_root', type=str, required=True,
                    help='grid_root (a fs:// or hdfs://)')

if __name__ == '__main__':
    args = parser.parse_args()

    grid_root = args.grid_root

    header = jgrid.Header.load(grid_root)
    fracnums = header.list_available_fracnums()
    fractions = header.list_available_fractions()
    print 'Header : ', header
    print 'fractions shape : ', header.frac_height, header.frac_width
    print 'fractions count : ', len(fractions)
    print 'fracnum counts : ', len(fracnums)
