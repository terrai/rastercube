"""
Script to download NDVI HDF files

A few notes on how this works :

The MODIS website we use is :

    http://e4ftl01.cr.usgs.gov/MOLT/MOD13Q1.005/ (for Terra)
    http://e4ftl01.cr.usgs.gov/MOLT/MOD13Q1.005/ (for Aqua)

On the MODIS website, each date has a HTML files that list the available HDF
files. We build a cache of the HTML for each date so we don't have to fetch
it every time (since this is *REALLY* slow)

This cache is saved in --modis_cache_dir (usually
$RASTERCUBE_DATA/0_input/modis_www_cache)

Then, we have to parse each date HTML file to get the list of available HDF
files and download the HDF files that are not present on the filesystem.
"""
#import requests_cache
#requests_cache.install_cache()

import os
import argparse
from datetime import datetime
import calendar
import rastercube.utils as utils
import rastercube.config as config
import rastercube.datasources.modis as modis
import urlparse
import cPickle as pickle
import tempfile
import shutil
import time
import re
import requests
import urllib
import tempfile
import subprocess
import shutil
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description='Create NDVI/QA jgrids from HDF')
parser.add_argument('--hdfdir', type=str, required=False, default=None,
                    help='source directory containing '
                         'HDF files, organised in per-year subdirectories'
                         '(e.g. $RASTERCUBE_DATA/0_input/MOD13Q1.005/HDF/LAT/)')
parser.add_argument('--skip_mirror', action='store_true',
                    help='Skip the mirroring, just do the download')
parser.add_argument('--modis_mirror_dir', type=str, default=None,
                    help='Directory where to mirror MODIS html files for each '
                         ' date')
parser.add_argument('--clear_filelist_cache', action='store_true',
                    help='Clear filelist cache prior to download')
parser.add_argument('--tile', default=None,
                    help='Download a specific tile (disregarding ' +
                         'config.MODIS_TILES')
parser.add_argument('--force_wget', type=bool, default=False,
                    help='If true, forces the use of the sequencial wget')
parser.add_argument('--terra_only', type=bool, default=False,
                    help='If true, will only download terra files (not aqua)')
parser.add_argument('--verbose', type=bool, default=False,
                    help='If true, increases verbosity')


DATE_REGEX = r'\d{4}\.\d{2}\.\d{2}'


def extract_dates_from_modis_index(contents):
    """
    Returns all the dates links from the MODIS index page listing all the
    dates directories
    """
    matcher = re.compile(DATE_REGEX)
    soup = BeautifulSoup(contents, 'html.parser')
    dates = [l.get('href') for l in soup.find_all('a')]
    dates = filter(lambda s: matcher.match(s) is not None, dates)
    return dates


def collect_all_dates_pages(url):
    """
    Collect the URL of all the dates pages given the base MODIS url
    """
    r = requests.get(url)
    dates = extract_dates_from_modis_index(r.text)
    return [(d.strip('/'), urlparse.urljoin(url, d)) for d in dates]


def parse_available_hdf_for_date(dateurl, contents):
    """
    Collect all the hdf files for a given date page
    Returns a list of tuples
        (tile_name, timestamp_ms, filename, full_url)
    """
    p = re.compile(modis.MODIS_NDVI_REGEX)

    # BeautifulSoup fails to find some <a> (probably because of messy
    # encoding). So fallback on regexp
    #soup = BeautifulSoup(contents, 'lxml')
    #print contents
    #print len(soup.find_all('a')), ' links found'
    #all_files = [l.get('href').strip() for l in soup.find_all('a')]

    LINK_REGEXP = r'\<a\s+href="(?P<href>[\w\.]+)"\>'
    a_regexp = re.compile(LINK_REGEXP)
    all_files = a_regexp.findall(contents)

    hdf_files = []
    for f in all_files:
        m = p.match(f.lower())
        if m:
            try:
                tile_name = 'h' + m.group('tile_h') + 'v' + m.group('tile_v')
                coll = m.group('coll')
                # Not sure if we handle other coll correctly, so fail fast
                assert coll == '005'
                julian_date = m.group('julian_date')
                date = datetime.strptime(julian_date, '%Y%j')
                timestamp_ms = int(calendar.timegm(date.timetuple()) * 1000.0)
                full_url = urlparse.urljoin(dateurl, f)
                hdf_files.append((tile_name, timestamp_ms, f, full_url))
            except Exception as e:
                print 'Failed for file %s' % f
                raise e
    return hdf_files


def collect_available_hdf_from_mirror(mirror_dir, base_url):
    """
    Loads all the HTML in the mirror dir and build a list of all available
    HDF files.

    Returns a list of tuples
        (tile_name, timestamp_ms, filename, full_url)
    """
    p = re.compile(r'([\d\.]+)\.html')
    all_hdf_files = []
    for fname in os.listdir(mirror_dir):
        if not fname.endswith('.html'):
            continue
        m = p.match(fname)
        assert m is not None
        date = m.group(1)
        dateurl = urlparse.urljoin(base_url, date) + '/'
        with open(os.path.join(mirror_dir, fname)) as f:
            contents = f.read()
        contents = contents.decode('ISO-8859-1').encode('utf-8')
        all_hdf_files += parse_available_hdf_for_date(dateurl, contents)
    return all_hdf_files


def mirror_modis_dates_html(base_url, mirror_dir):
    """
    Download all MODIS date listing pages to a local directory.
    Usually, a MODIS listing for a date should not change (only new dates
    should be added), so there should be no need to re-download.
    """
    ndownloads = 0
    dates_urls = collect_all_dates_pages(base_url)
    utils.mkdir_p(mirror_dir)
    for date, url in dates_urls:
        fname = os.path.join(mirror_dir, date + '.html')
        if not os.path.exists(fname):
            print 'Downloading ', fname
            urllib.urlretrieve(url, fname)
            ndownloads += 1
            # The MODIS MOLT repository server doesn't return Content-Length
            # so urllib cannot tell if it downloaded the whole html or was
            # just disconnected, which could lead to incomplete HTML being
            # downloaded. So we check if the downloaded file ends with </html>
            with open(fname, 'r') as f:
                # seek 10 bytes from the end
                f.seek(-10, 2)
                line = f.read(10)
                if not "</html>" in line:
                    raise urllib.ContentTooShortError("Couldn't find </html>" +
                            " in downloaded file, probably a partial download")

            # Just avoid firing requests as fast as possible
            time.sleep(0.1)

    return ndownloads > 0


def download_url(url, dst_filename):
    """
    Download url into dst_filename.
    To avoid having half-complete files lying around, this first downloads
    to a temporary location and move to dst_filename once the download
    is complete
    """
    print 'Starting ', dst_filename
    # Ensure year directory exists
    year_dir = os.path.join(os.path.dirname(dst_filename), os.pardir)
    utils.mkdir_p(year_dir)
    with tempfile.NamedTemporaryFile() as f:
        subprocess.check_call('/usr/bin/wget %s -O %s' % (url, f.name),
                              shell=True)
        shutil.copyfile(f.name, dst_filename)
    print 'Finished ', dst_filename


def download_files_wget(missing_files):
    for url, dst_filename in missing_files:
        download_url(url, dst_filename)


def _do_download_aria2(files):
    # We download to a temporary loaction and copy to final destination once
    # finished
    try:
        tmpdir = tempfile.mkdtemp()
        print '==== Downloading to tmpdir=%s' % tmpdir
        # Create temporary filenames and aria2 download file
        download_fname = os.path.join(tmpdir, 'downloads.txt')
        tmp2fname = {}
        with open(download_fname, 'w') as f:
            for i in range(len(files)):
                url, dst_filename = files[i]
                tmpname = os.path.join(tmpdir, '%d' % i)
                tmp2fname[tmpname] = dst_filename

                # Read aria2 docs for the format of the input file. Basically,
                # an URI line can be followed by options line which MUST start
                # with one or more spaces
                # https://aria2.github.io/manual/en/html/aria2c.html
                # This is also relevant with some tips for how to write the
                # file https://github.com/tatsuhiro-t/aria2/issues/190
                f.write('%s\n out=%s\n' % (url, tmpname))

                del url, dst_filename

        # Download using aria2
        cmd = ['/usr/bin/aria2c', '-i %s' % download_fname]
        # Set the cwd to / because aria2 interpret filenames as relative
        # even if they start with /
        subprocess.check_call(cmd, cwd='/')

        # Copy files to their final destination
        for tmpname, dstname in tmp2fname.items():
            shutil.copyfile(tmpname, dstname)
            print 'Finished ', dstname

    finally:
        shutil.rmtree(tmpdir)


def download_files_aria2(missing_files):
    # download in chunks
    nfiles = len(missing_files)
    # We get blacklisted if we try to download more at once
    chunksize = 4
    for i in range(0, nfiles, chunksize):
        files = missing_files[i:min(i+chunksize, nfiles)]
        _do_download_aria2(files)


# Use a cache
MODIS_HDF_CACHE = '/tmp/modis_cache.pickle'


if __name__ == '__main__':
    args = parser.parse_args()

    if args.clear_filelist_cache:
        if os.path.exists(MODIS_HDF_CACHE):
            os.unlink(MODIS_HDF_CACHE)

    hdf_dir = args.hdfdir
    if hdf_dir is None:
        hdf_dir = utils.get_modis_hdf_dir()

    root_mirror_dir = args.modis_mirror_dir
    if root_mirror_dir is None:
        root_mirror_dir = os.path.join(utils.get_data_dir(), '0_input',
                                       'modis_www_mirror')
    print 'Using MODIS cache directory'

    existing_hdf_files = modis.ndvi_list_hdf(hdf_dir)

    if args.tile is not None:
        terra_tiles = set([args.tile])
        aqua_tiles = set([args.tile])
    else:
        terra_tiles = set(config.MODIS_TERRA_TILES)
        aqua_tiles = set(config.MODIS_AQUA_TILES)

    print 'MODIS Terra URL : ', config.MODIS_TERRA_URL
    print 'MODIS Aqua URL : ', config.MODIS_AQUA_URL
    print 'Terra tiles ', ' '.join(terra_tiles)
    print 'Aqua tiles ', ' '.join(aqua_tiles)

    # -- Mirror aqua and terra HTML file lists
    terra_mirror_dir = os.path.join(root_mirror_dir, 'MOD13Q1')
    aqua_mirror_dir = os.path.join(root_mirror_dir, 'MYD13Q1')
    if not args.skip_mirror:
        nchanged = 0
        nchanged += mirror_modis_dates_html(
            config.MODIS_TERRA_URL, terra_mirror_dir)
        nchanged += mirror_modis_dates_html(
            config.MODIS_AQUA_URL, aqua_mirror_dir)

        if nchanged > 0:
            # Some new dates have been downloaded, clear the HDF cache we had
            if os.path.exists(MODIS_HDF_CACHE):
                os.unlink(MODIS_HDF_CACHE)

    # -- Load the list of available HDF from the mirror directory
    if not os.path.exists(MODIS_HDF_CACHE):
        print 'Building MODIS HDF cache'
        all_hdf_files = {
            'terra' : collect_available_hdf_from_mirror(
                terra_mirror_dir, config.MODIS_TERRA_URL
            ),
            'aqua' : collect_available_hdf_from_mirror(
                aqua_mirror_dir, config.MODIS_AQUA_URL
            ),
        }
        with open(MODIS_HDF_CACHE, 'w') as f:
            pickle.dump(all_hdf_files, f)
    else:
        with open(MODIS_HDF_CACHE) as f:
            all_hdf_files = pickle.load(f)

    print '-- Available HDF files'
    print 'terra :', len(all_hdf_files['terra'])
    print 'aqua :', len(all_hdf_files['aqua'])

    # -- Build the list of files to download
    print '-- Files to download'
    missing_files = []

    steps = [('terra', terra_tiles)]
    if not args.terra_only:
        steps.append(('aqua', aqua_tiles))

    for satname, tiles in steps:
        remote_hdf_files = all_hdf_files[satname]

        # At this point, remote_hdf_files contains a list of tuples
        # (tile_name, timestampms, fname, url)

        for tile_name, timestampms, fname, url in remote_hdf_files:
            if tile_name in tiles:
                file_date = utils.date_from_timestamp_ms(timestampms)
                yearstr = str(file_date.year)
                full_fname = os.path.join(hdf_dir, yearstr, fname)
                if not os.path.exists(full_fname):
                    missing_files.append((url, full_fname))
                    if args.verbose:
                        print file_date, tile_name, '=>', full_fname

    print len(missing_files), ' files to download'
    print '\n\n-- Starting downloads'

    # Download the files
    if args.force_wget:
        download_files_wget(missing_files)
    else:
        download_files_aria2(missing_files)

    print 'Finished download'
