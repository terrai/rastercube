import os

HDFS_MASTER = 'IICT-SV2455.lan.iict.ch'
SPARK_MASTER = os.environ.get('SPARK_MASTER',
                              'spark://iict-sv2455.lan.iict.ch:7077')

MODIS_TERRA_URL = 'http://e4ftl01.cr.usgs.gov/MOLT/MOD13Q1.005/'
MODIS_AQUA_URL = 'http://e4ftl01.cr.usgs.gov/MOLA/MYD13Q1.005/'

MODIS_TERRA_TILES = [
    # central america
    'h09v07',
    # south america
    'h10v08', 'h11v08', 'h12v08', 'h10v09', 'h11v09', 'h12v09',
    'h13v09', 'h14v09', 'h12v10', 'h13v10',
    'h12v11', 'h13v11', 'h12v12', 'h13v12',
    'h13v13', 'h09v09', 'h10v10', 'h11v10', 'h12v10', 'h14v10',
    'h11v11', 'h12v13',
    # africa
    'h16v07', 'h16v08', 'h18v07', 'h19v07', 'h20v07', 'h21v07',
    'h17v07', 'h17v08', 'h18v08', 'h19v08', 'h20v08', 'h21v08',
    'h19v09', 'h20v09', 'h21v09',
    # asia
    'h26v06', 'h27v06', 'h28v06',
    'h25v07', 'h27v07', 'h28v07', 'h29v07', 'h30v07',
    'h27v08', 'h28v08', 'h29v08', 'h30v08',
    'h28v09', 'h29v09', 'h30v09', 'h31v09', 'h32v09',
]
MODIS_AQUA_TILES = list(MODIS_TERRA_TILES)
